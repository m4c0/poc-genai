#pragma leco app
#include <stdio.h>

import dotz;
import gpu;
import jason;
import jojo;
import jute;
import hai;
import print;
import traits;

using namespace traits::ints;
namespace j = jason::ast;
namespace jn = j::nodes;

template<unsigned W, unsigned H>
struct f32a {
  hai::array<float> data { W * H };
};
template<unsigned W, unsigned H>
struct f32v {
  const float * data {};

  f32v() = default;
  explicit f32v(const float * d) : data { d } {}
  f32v(f32a<W, H> & a) : data { a.data.begin() } {}
};

static constexpr const auto n_ctx = 1024;
static constexpr const auto n_embed = 768;
static constexpr const auto n_eps = 1e-05;
static constexpr const auto n_head = 12;
static constexpr const auto n_layer = 12;
static constexpr const auto n_vocab = 50257;

static constexpr const auto emb_hd = n_embed / n_head;

static jute::view g_cnt {};
static const jn::dict * g_config {};

static hai::uptr<gpu> g_gpu {};

#if 0
static void debug(const float *xx, int r, int c) {
  for (auto i = 0; i < r; i++) {
    if (i >= 3 && i < r - 3) {
      if (i == 3) putln("...");
      continue;
    }
    for (auto j = 0; j < c; j++) {
      float x = xx[i * c + j];
      if (j < 3) putf("%+.4f ", x);
      else if (j > c - 3) putf("%+.4f ", x);
      else if (j == 4) put("... ");
    }
    putln();
  }
  putln();
}
template<unsigned W, unsigned H>
static void debug(const f32a<W, H> & x) { debug(x.data.begin(), W, H); }
#endif

template<unsigned W, unsigned H>
static f32v<W, H> extract(jute::view key) {
  auto & root = *g_config;
  auto & v = j::cast<jn::dict>(root[key]);
  auto dtype = j::cast<jn::string>(v["dtype"]).str();
  if (*dtype != "F32") die("unsupported dtype ", *dtype);

  auto & shape = j::cast<jn::array>(v["shape"]);
  hai::array<int> shp { shape.size() };
  for (auto i = 0; i < shp.size(); i++) {
    shp[i] = j::cast<jn::number>(shape[i]).integer();
  }

  auto & offs = j::cast<jn::array>(v["data_offsets"]);
  auto start = j::cast<jn::number>(offs[0]).integer();
  auto end = j::cast<jn::number>(offs[1]).integer();
  if (end < start || end - start > g_cnt.size()) die("invalid offsets ", start, "~", end);

  // unsigned len = end - start;
  return f32v<W, H>{ reinterpret_cast<const float *>(g_cnt.begin() + start) };
}
template<unsigned W, unsigned H>
static auto extract(int layer, const char * a, const char * b) {
  char buf[1024] {};
  snprintf(buf, sizeof(buf), "h.%d.%s.%s", layer, a, b);
  return extract<W, H>(jute::view::unsafe(buf));
}

static auto layer_norm(f32a<n_ctx, n_embed> & x, unsigned tks, f32v<n_embed, 1> weight, f32v<n_embed, 1> bias) {
  f32a<n_ctx, n_embed> res {};

  for (auto i = 0; i < tks; i++) {
    float mean {};
    float variance {};

    auto x_ptr = &x.data[i * n_embed];
    for (auto j = 0; j < n_embed; j++) {
      mean += x_ptr[j];
    }
    mean /= n_embed;

    for (auto j = 0; j < n_embed; j++) {
      auto d = x_ptr[j] - mean;
      variance += d * d;
    }
    variance /= n_embed;
    variance += n_eps;

    // g * x + b
    auto res_ptr = &res.data[i * n_embed];
    for (auto j = 0; j < n_embed; j++) {
      res_ptr[j] = weight.data[j] * (x_ptr[j] - mean) / dotz::sqrt(variance) + bias.data[j];
    }
  }

  return res;
}
static auto layer_norm(f32a<n_ctx, n_embed> & x, unsigned tks, int layer, const char * ln) {
  auto bias = extract<n_embed, 1>(layer, ln, "bias");
  auto weight = extract<n_embed, 1>(layer, ln, "weight");
  return layer_norm(x, tks, weight, bias);
}

// x @ w + b -- cij = aik * bkj
template<unsigned I, unsigned J, unsigned K>
static auto linear(f32a<I, K> & x, int layer, const char * mats, f32v<I, J> init) {
  auto w = extract<K, J>(layer, mats, "weight");
  auto b = extract<J, 1>(layer, mats, "bias");
  f32a<I, J> res {};

#if 1
  g_gpu->load(0, [&](auto * p) {
    for (auto i = 0; i < I * K; i++, p++) *p = x.data[i];
  });
  g_gpu->load(1, [&](auto * p) {
    for (auto i = 0; i < K * J; i++, p++) *p = w.data[i];
  });
  g_gpu->load(2, [&](auto * p) {
    for (auto j = 0; j < J; j++, p++) *p = b.data[j];
  });

  g_gpu->run(I, J, K);

  g_gpu->load(3, [&](auto * p) {
    for (auto ij = 0; ij < I * J; ij++, p++) {
      auto i = init.data ? init.data[ij] : 0;
      res.data[ij] = *p + i;
    }
  });
#else
  auto ptr = res.data.begin();
  for (auto i = 0; i < I; i++) {
    auto x_ptr = &x.data[i * K];
    for (auto j = 0; j < J; j++, ptr++) {
      *ptr = init.data ? init.data[i * J + j] : 0;
      *ptr += b.data[j];
      for (auto k = 0; k < K; k++) {
        *ptr += x_ptr[k] * w.data[k * J + j];
      }
    }
  }
#endif
  return res;
}

static auto mha(f32a<n_ctx, n_embed> & x, unsigned tks, int layer) {
  auto xn = layer_norm(x, tks, layer, "ln_1");
  auto qkv = linear<n_ctx, n_embed * 3, n_embed>(xn, layer, "attn.c_attn", {});

  f32a<n_ctx, n_embed> hstack {};
  f32a<n_ctx, n_ctx> smax {};
  for (auto h = 0; h < n_head; h++) {
    // q @ k.T / sqrt(tks) + mask
    auto smax_ptr = smax.data.begin();
    for (auto i = 0; i < tks; i++) {
      auto q_ptr = &qkv.data[i * n_embed * 3 + h * emb_hd];
      for (auto j = 0; j < tks; j++, smax_ptr++) {
        auto k_ptr = &qkv.data[j * n_embed * 3 + n_embed + h * emb_hd];
        *smax_ptr = 0;
        for (auto k = 0; k < emb_hd; k++) {
          // j/k in "k" flipped to compensate k.T
          *smax_ptr += q_ptr[k] * k_ptr[k];
        }
        *smax_ptr /= dotz::sqrt(static_cast<float>(emb_hd));
        if (j > i) *smax_ptr += -1e10; // mask
      }
    }
    // softmax
    for (auto i = 0; i < tks; i++) {
      float max = -1e10;
      for (auto j = 0; j < tks; j++) {
        auto sij = smax.data[i * tks + j];
        if (sij > max) max = sij;
      }

      float sum = 0;
      for (auto j = 0; j < tks; j++) {
        auto & sij = smax.data[i * tks + j];
        sij = dotz::exp(sij - max);
        sum += sij;
      }

      for (auto j = 0; j < tks; j++) {
        auto & sij = smax.data[i * tks + j];
        sij /= sum;
      }
    }

    // smax @ v
    for (auto i = 0; i < tks; i++) {
      for (auto j = 0; j < emb_hd; j++) {
        auto hstack_ptr = &hstack.data[i * n_embed + h * emb_hd + j];
        *hstack_ptr = 0;
        for (auto k = 0; k < tks; k++) {
          auto v_ptr = &qkv.data[k * n_embed * 3 + 2 * n_embed + h * emb_hd];
          *hstack_ptr += smax.data[i * tks + k] * v_ptr[j];
        }
      }
    }
  }

  return linear<n_ctx, n_embed, n_embed>(hstack, layer, "attn.c_proj", x);
}

static constexpr const auto pi = 3.14159265358979323;
static auto ffn(f32a<n_ctx, n_embed> & x, int tks, int layer) {
  auto xn = layer_norm(x, tks, layer, "ln_2");
  auto a = linear<n_ctx, n_embed * 4, n_embed>(xn, layer, "mlp.c_fc", {});
  for (auto & f : a.data) {
    using namespace dotz;
    f = 0.5 * f * (1 + tanh(sqrt(2.0 / pi) * (f + 0.044715 * f * f * f)));
  }

  return linear<n_ctx, n_embed, n_embed * 4>(a, layer, "mlp.c_proj", x);
}

static void print_token(const auto & vocab, int tk) {
  putf("%5d ", tk);
  if (tk == 198) {
    putln();
    return;
  }
  for (auto &[txt, id]: vocab) {
    auto id_i = j::cast<jn::number>(id).integer();
    if (id_i != tk) continue;
    putln(txt);
    return;
  }
  putln();
}

int main(int argc, char ** argv) try {
  // use the largest size required
  g_gpu = hai::uptr<gpu>::make(static_cast<unsigned>(n_embed * n_vocab * sizeof(float)));

  auto vocab_file = jojo::read_cstr("out/vocab.json");
  auto vocab_json = jason::parse(vocab_file);
  auto & vocab = j::cast<jn::dict>(vocab_json);

  auto model_raw = jojo::read("out/model.safetensors");
  jute::view model { model_raw.begin(), model_raw.size() };

  auto hdr_size = *reinterpret_cast<const uint64_t *>(model.begin());
  auto [sz, hdr, cnt] = model.subview(8, hdr_size);
  g_cnt = cnt;

  if (hdr_size != hdr.size())
    die("invalid safetensor - expecting header with size ", hdr_size, ", got ", hdr.size());
  
  auto json = jason::parse(hdr);
  g_config = &j::cast<jn::dict>(json);

  hai::array<unsigned> in_tks { n_ctx };

  // TODO: implement the encoder properly
  unsigned tks = 0;
  in_tks[tks++] = j::cast<jn::number>(vocab["Who"]).integer();
  in_tks[tks++] = j::cast<jn::number>(vocab["Ġis"]).integer();
  in_tks[tks++] = j::cast<jn::number>(vocab["ĠAlfred"]).integer();
  in_tks[tks++] = j::cast<jn::number>(vocab["ĠNobel"]).integer();
  in_tks[tks++] = j::cast<jn::number>(vocab["?"]).integer();

  for (auto i = 0; i < tks; i++) print_token(vocab, in_tks[i]);

  // TODO: assert wpe/wte sizes
  auto wte = extract<n_vocab, n_embed>("wte.weight");
  auto wpe = extract<n_ctx, n_embed>("wpe.weight");
  auto lnf_b = extract<n_embed, 1>("ln_f.bias");
  auto lnf_w = extract<n_embed, 1>("ln_f.weight");

  for (; tks < in_tks.size(); tks++) {
    f32a<n_ctx, n_embed> x {};

    // x = wte[token_ids] + wpe[[0, 1, 2...]]
    for (auto i = 0; i < tks; i++) {
      auto wte_ptr = &wte.data[in_tks[i] * n_embed];
      auto wpe_ptr = &wpe.data[i * n_embed];
      auto x_ptr   = &x.data[i * n_embed];
      for (auto j = 0; j < n_embed; j++) {
        x_ptr[j] = wte_ptr[j] + wpe_ptr[j];
      }
    }

    for (auto i = 0; i < n_layer; i++) {
      auto m = mha(x, tks, i);
      x = ffn(m, tks, i);
    }

    auto xn = layer_norm(x, tks, lnf_w, lnf_b);

    // argmax((xn @ wte.T)[-1])
    float max = -1e10;
    int v_id = -1;
    for (auto i = tks - 1; i < tks; i++) {
      auto x_ptr = &xn.data[i * n_embed];
      for (auto j = 0; j < n_vocab; j++) {
        float n = 0;
        for (auto k = 0; k < n_embed; k++) {
          n += x_ptr[k] * wte.data[j * n_embed + k];
        }
        if (n > max) {
          max = n;
          v_id = j;
        }
      }
    }

    in_tks[tks] = v_id;
    print_token(vocab, v_id);
  }

  putln();
} catch (...) {
  return 1;
}
