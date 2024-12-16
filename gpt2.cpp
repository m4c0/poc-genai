#pragma leco tool
#include <stdio.h>

import dotz;
import jason;
import jojo;
import jute;
import hai;
import print;
import traits;

using namespace traits::ints;
namespace j = jason::ast;
namespace jn = j::nodes;

using f32a = hai::array<float>;

static constexpr const auto n_ctx = 1024;
static constexpr const auto n_embed = 768;
static constexpr const auto n_eps = 1e-05;
static constexpr const auto n_head = 12;
static constexpr const auto n_layer = 12;
static constexpr const auto n_vocab = 50257;

static constexpr const auto emb_hd = n_embed / n_head;

static jute::view g_cnt {};
static const jn::dict * g_config {};

#if 0
static void debug(const float *xx, int r, int c) {
  for (auto i = 0; i < r; i++) {
    if (i >= 3 && i < r - 3) {
      if (i == 3) putln("...");
      continue;
    }
    for (auto j = 0; j < c; j++, xx++) {
      if (j < 3) put(*xx, " ");
      else if (j > c - 3) put(*xx, " ");
      else if (j == 4) put("... ");
    }
    putln();
  }
  putln();
}
static void debug(const f32a & x, int r, int c) { debug(x.begin(), r, c); }
#endif

static const float * extract(jute::view key) {
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
  return reinterpret_cast<const float *>(g_cnt.begin() + start);
}
static auto extract(int layer, const char * a, const char * b) {
  char buf[1024];
  snprintf(buf, sizeof(buf), "h.%d.%s.%s", layer, a, b);
  return extract(jute::view::unsafe(buf));
}

static auto layer_norm(f32a & x, unsigned tks, const float * weight, const float * bias) {
  f32a res { x.size() };

  for (auto i = 0; i < tks; i++) {
    float mean {};
    float variance {};

    auto x_ptr = &x[i * n_embed];
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

    auto res_ptr = &res[i * n_embed];
    for (auto j = 0; j < n_embed; j++) {
      res_ptr[j] = weight[j] * (x_ptr[j] - mean) / dotz::sqrt(variance) + bias[j];
    }
  }

  return res;
}
static auto layer_norm(f32a & x, unsigned tks, int layer, const char * ln) {
  auto bias = extract(layer, ln, "bias");
  auto weight = extract(layer, ln, "weight");
  return layer_norm(x, tks, weight, bias);
}

// x @ w + b;
static auto linear(f32a & x, unsigned mi, unsigned mj, unsigned mk, int layer, const char * mats, const float * init) {
  auto w = extract(layer, mats, "weight");
  auto b = extract(layer, mats, "bias");

  f32a res { mi * mj };
  auto ptr = res.begin();
  for (auto i = 0; i < mi; i++) {
    auto x_ptr = &x[i * mk];
    for (auto j = 0; j < mj; j++, ptr++) {
      *ptr = init ? init[i * mk + j] : 0;
      *ptr += b[j];
      for (auto k = 0; k < mk; k++) {
        *ptr += x_ptr[k] * w[k * mj + j];
      }
    }
  }
  return res;
}

static auto mha(f32a & x, unsigned tks, int layer) {
  auto xn = layer_norm(x, tks, layer, "ln_1");
  auto qkv = linear(xn, tks, n_embed * 3, n_embed, layer, "attn.c_attn", nullptr);

  f32a hstack { tks * n_embed };
  f32a smax { tks * tks };
  for (auto h = 0; h < n_head; h++) {
    // q @ k.T / sqrt(tks) + mask
    auto smax_ptr = smax.begin();
    for (auto i = 0; i < tks; i++) {
      auto q_ptr = &qkv[i * n_embed * 3 + h * emb_hd];
      for (auto j = 0; j < tks; j++, smax_ptr++) {
        auto k_ptr = &qkv[j * n_embed * 3 + n_embed + h * emb_hd];
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
        auto sij = smax[i * tks + j];
        if (sij > max) max = sij;
      }

      float sum = 0;
      for (auto j = 0; j < tks; j++) {
        auto & sij = smax[i * tks + j];
        sij = dotz::exp(sij - max);
        sum += sij;
      }

      for (auto j = 0; j < tks; j++) {
        auto & sij = smax[i * tks + j];
        sij /= sum;
      }
    }

    // smax @ v
    for (auto i = 0; i < tks; i++) {
      for (auto j = 0; j < emb_hd; j++) {
        auto hstack_ptr = &hstack[i * n_embed + h * emb_hd + j];
        *hstack_ptr = 0;
        for (auto k = 0; k < tks; k++) {
          auto v_ptr = &qkv[k * n_embed * 3 + 2 * n_embed + h * emb_hd];
          *hstack_ptr += smax[i * tks + k] * v_ptr[j];
        }
      }
    }
  }

  return linear(hstack, tks, n_embed, n_embed, layer, "attn.c_proj", x.begin());
}

static constexpr const auto pi = 3.14159265358979323;
static auto ffn(f32a & x, int tks, int layer) {
  auto a = linear(x, tks, n_embed * 4, n_embed, layer, "mlp.c_fc", nullptr);
  for (auto & f : a) {
    using namespace dotz;
    f = 0.5 * f * (1 + tanh(sqrt(2.0 / pi) * (f + 0.044715 * f * f *f)));
  }

  return linear(a, tks, n_embed, n_embed * 4, layer, "mlp.c_proj", x.begin());
}

int main(int argc, char ** argv) try {
  if (argc != 2) die("missing safetensor filename");

  auto model_raw = jojo::read(jute::view::unsafe(argv[1]));
  jute::view model { model_raw.begin(), model_raw.size() };

  auto hdr_size = *reinterpret_cast<const uint64_t *>(model.begin());
  auto [sz, hdr, cnt] = model.subview(8, hdr_size);
  g_cnt = cnt;

  if (hdr_size != hdr.size())
    die("invalid safetensor - expecting header with size ", hdr_size, ", got ", hdr.size());
  
  auto json = jason::parse(hdr);
  g_config = &j::cast<jn::dict>(json);

  // TODO: use a string encoder
  // "Hello there!" plus space to generate more tokens
  auto in_tks = hai::array<unsigned>::make(15496, 612, 5145, 0, 0, 0);
  auto tks = 3;

  f32a x { n_ctx * n_embed };

  // TODO: assert wpe/wte sizes
  auto wte = extract("wte.weight");
  auto wpe = extract("wpe.weight");

  // x = wte[token_ids] + wpe[[0, 1, 2...]]
  for (auto i = 0; i < tks; i++) {
    auto wte_ptr = &wte[in_tks[i] * n_embed];
    auto wpe_ptr = &wpe[i * n_embed];
    auto x_ptr   = &x[i * n_embed];
    for (auto j = 0; j < n_embed; j++) {
      x_ptr[j] = wte_ptr[j] + wpe_ptr[j];
    }
  }

  for (auto i = 0; i < n_layer; i++) {
    auto m = mha(x, tks, i);
    x = ffn(m, tks, i);
  }

  auto bias = extract("ln_f.bias");
  auto weight = extract("ln_f.weight");
  auto xn = layer_norm(x, tks, weight, bias);

  // argmax((xn @ wte.T)[-1])
  float max = -1e10;
  int v_id = -1;
  for (auto i = tks - 1; i < tks; i++) {
    auto x_ptr = &xn[i * n_embed];
    for (auto j = 0; j < n_vocab; j++) {
      float n = 0;
      for (auto k = 0; k < n_embed; k++) {
        n += x_ptr[k] * wte[j * n_embed + k];
      }
      if (n > max) {
        max = n;
        v_id = j;
      }
    }
  }

  putln(v_id);
} catch (...) {
  return 1;
}
