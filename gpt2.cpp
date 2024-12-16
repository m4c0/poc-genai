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
static constexpr const auto n_layer = 12;

static jute::view g_cnt {};
static const jn::dict * g_config {};

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
static void debug(const f32a & x, int r, int c) {
  debug(x.begin(), r, c);
}
static void debug_x(const f32a & x, int tks) { debug(x, tks, n_embed); }

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

static void init_x(f32a & x, const auto & in_tks) {
  // TODO: assert wpe/wte sizes
  auto wte = extract("wte.weight");
  auto wpe = extract("wpe.weight");

  // x = wte[token_ids] + wpe[[0, 1, 2...]]
  for (auto i = 0; i < in_tks.size(); i++) {
    auto wte_ptr = &wte[in_tks[i] * n_embed];
    auto wpe_ptr = &wpe[i * n_embed];
    auto x_ptr   = &x[i * n_embed];
    for (auto j = 0; j < n_embed; j++) {
      x_ptr[j] = wte_ptr[j] + wpe_ptr[j];
    }
  }
}

static auto layer_norm(f32a & x, unsigned tks, int layer, const char * ln) {
  auto bias = extract(layer, ln, "bias");
  auto weight = extract(layer, ln, "weight");

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

static auto mha(f32a & x, unsigned tks, int layer) {
  auto xn = layer_norm(x, tks, layer, "ln_1");
  auto w = extract(layer, "attn.c_attn", "weight");
  auto b = extract(layer, "attn.c_attn", "bias");

  // q+k+v for each embed
  f32a qkv { x.size() * 3 };
  // x @ w + b;
  auto qkv_ptr = qkv.begin();
  for (auto i = 0; i < tks; i++) {
    auto x_ptr = &xn[i * n_embed];
    for (auto j = 0; j < 3 * n_embed; j++, qkv_ptr++) {
      *qkv_ptr = b[j];
      for (auto k = 0; k < n_embed; k++) {
        *qkv_ptr += x_ptr[k] * w[k * 3 * n_embed + j];
      }
    }
  }

  f32a stfm { tks * tks };
  // q @ k.T / sqrt(tks) + mask
  auto stfm_ptr = stfm.begin();
  for (auto i = 0; i < tks; i++) {
#error This does not split QKV per n_head
    // FIXME: it is missing another level of loop of 1..n_head
    auto q_ptr = &qkv[i * n_embed * 3];
    for (auto j = 0; j < tks; j++, stfm_ptr++) {
      auto k_ptr = &qkv[j * n_embed * 3 + n_embed];
      *stfm_ptr = 0;
      for (auto k = 0; k < n_embed; k++) {
        // j/k in "k" flipped to compensate k.T
        *stfm_ptr += q_ptr[k] * k_ptr[k];
      }
      *stfm_ptr /= dotz::sqrt(static_cast<float>(n_embed));
      if (j > i) *stfm_ptr += -1e10; // mask
    }
  }

  for (auto i = 0; i < tks; i++) {
    float max = -1e10;
    for (auto j = 0; j < tks; j++) {
      auto sij = stfm[i * tks + j];
      if (sij > max) max = sij;
    }

    float sum = 0;
    for (auto j = 0; j < tks; j++) {
      auto & sij = stfm[i * tks + j];
      sij = dotz::exp(sij - max);
      sum += sij;
    }

    for (auto j = 0; j < tks; j++) {
      auto & sij = stfm[i * tks + j];
      sij /= sum;
    }
  }

  debug(stfm, tks, tks);
}

static void transform(f32a & x, int tks, int layer) {
  mha(x, tks, layer);
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
  // TODO: use real tokens
  auto in_tks = hai::array<unsigned>::make(1, 2, 3, 4, 5, 6);

  f32a x { n_ctx * n_embed };
  init_x(x, in_tks);
  for (auto i = 0; i < n_layer; i++) {
    transform(x, in_tks.size(), i);
    break;
  }

  debug_x(x, in_tks.size());
} catch (...) {
  return 1;
}
