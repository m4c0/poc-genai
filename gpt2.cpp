#pragma leco tool

import jason;
import jojo;
import jute;
import hai;
import print;
import traits;

using namespace traits::ints;
namespace j = jason::ast;
namespace jn = j::nodes;

static constexpr const auto n_ctx = 1024;
static constexpr const auto n_embed = 768;
static constexpr const auto n_layer = 12;

static jute::view g_cnt {};
static const jn::dict * g_config {};

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

static void init_x(hai::array<float> & x, const auto & in_tks) {
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

  hai::array<float> x { n_ctx * n_embed };
  init_x(x, in_tks);

  auto xx = x.begin();
  for (auto i = 0; i < in_tks.size(); i++) {
    for (auto j = 0; j < n_embed; j++, xx++) {
      if (j < 3) put(*xx, " ");
      if (j == 4) put("... ");
      if (j > n_embed - 3) put(*xx, " ");
    }
    putln();
  }
} catch (...) {
  return 1;
}
