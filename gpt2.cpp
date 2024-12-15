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

static const float * extract(const auto & json, jute::view key, jute::view cnt) {
  auto & root = j::cast<jn::dict>(json);
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
  if (end < start || end - start > cnt.size()) die("invalid offsets ", start, "~", end);

  // unsigned len = end - start;
  return reinterpret_cast<const float *>(cnt.begin() + start);
}

int main(int argc, char ** argv) try {
  if (argc != 2) die("missing safetensor filename");

  auto model_raw = jojo::read(jute::view::unsafe(argv[1]));
  jute::view model { model_raw.begin(), model_raw.size() };

  auto hdr_size = *reinterpret_cast<const uint64_t *>(model.begin());
  auto [sz, hdr, cnt] = model.subview(8, hdr_size);

  if (hdr_size != hdr.size())
    die("invalid safetensor - expecting header with size ", hdr_size, ", got ", hdr.size());
  
  auto json = jason::parse(hdr);

  // TODO: use a string encoder
  // TODO: use real tokens
  auto in_tks = hai::array<unsigned>::make(1, 2, 3, 4, 5, 6);

  auto wte = extract(json, "wte", cnt);
  auto wpe = extract(json, "wpe", cnt);

  hai::array<float> x { 1024 * 768}; // num_xxx * num_yyy
  for (auto i = 0; i < in_tks.size(); i++) {
    auto wte_ptr = &wte[in_tks[i] * 768];
    auto wpe_ptr = &wpe[i * 768];
    auto x_ptr   = &x[i * 768];
    for (auto j = 0; j < 768; j++) {
      x_ptr[j] = wte_ptr[j] + wpe_ptr[j];
    }
  }
} catch (...) {
  return 1;
}
