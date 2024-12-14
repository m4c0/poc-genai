#pragma leco tool

import jason;
import jojo;
import jute;
import print;
import traits;

using namespace traits::ints;

int main(int argc, char ** argv) try {
  if (argc != 2) die("missing filename");

  auto model_raw = jojo::read(jute::view::unsafe(argv[1]));
  jute::view model { model_raw.begin(), model_raw.size() };

  auto hdr_size = *reinterpret_cast<const uint64_t *>(model.begin());
  auto [sz, hdr, cnt] = model.subview(8, hdr_size);

  if (hdr_size != hdr.size())
    die("invalid safetensor - expecting header with size ", hdr_size, ", got ", hdr.size());
  
  auto json = jason::parse(hdr);
} catch (...) {
  return 1;
}
