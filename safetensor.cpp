#pragma leco tool

import jason;
import jojo;
import jute;
import hai;
import print;
import traits;

using namespace traits::ints;

static constexpr bool is_first_layer_key(jute::view v) {
  return v == "embed" || v == "wte" || v == "wpe" || v == "shared";
}
static constexpr bool is_last_layer_key(jute::view v) {
  return v == "head" || v == "classifier";
}
static constexpr int atoi(jute::view v) {
  int res = 0;
  for (auto c : v) {
    if (c < '0' || c > '9') return -1;
    res = res * 10 + (c - '0');
  }
  return res;
}
static constexpr int operator<=>(jute::view a, jute::view b) {
  for (auto i = 0; true; i++) {
    if (i >= a.size() && i >= b.size()) return 0;
    if (i >= a.size()) return -1;
    if (i >= b.size()) return 1;
    if (a[i] != b[i]) return a[i] - b[i];
  }
}
static constexpr int key_cmp(jute::view in_a, jute::view in_b) {
  // Copied from here: https://github.com/huggingface/safetensors/issues/44#issuecomment-1729647056
  //
  // 1. Split a layer name. The splitters/seperators are [".", "-", "_"].
  //    Example: h.0.attn.c_proj.bias -> ["h", 0, "attn", "c_proj", "bias"]
  // 2. Compare layername objects. If the current element is string, do
  //    lexiocographical order. If they are numbers, do numbers order.
  //    Ex: ["h", 0, "attn", "c_proj", "bias"] will rank higher than 
  //        ["h", 1, "attn", "c_proj", "bias"] because 0 < 1 in their second
  //        elements
  // 3. Use the below heauristic names/regexes (copied mostly from transformers
  //    naming convention), to "overwrite" the lexiocographical order
  //
  // const REGEX_FIRST_LAYERS = /(embed|wte|wpe|shared)/i;
  // const REGEX_LAST_LAYERS = /(head|classifier)/i;

  auto a = in_a;
  auto b = in_b;
  while (a.size() && b.size()) {
    auto [al, ar] = a.split('.');
    auto [bl, br] = b.split('.');

    auto af = is_first_layer_key(al);
    auto bf = is_first_layer_key(bl);
    if (af && !bf) return -1;
    if (!af && bf) return 1;

    af = is_last_layer_key(al);
    bf = is_last_layer_key(bl);
    if (af && !bf) return 1;
    if (!af && bf) return -1;

    int ai = atoi(al);
    int bi = atoi(bl);
    if (ai != -1 && bi != -1) {
      if (ai != bi) return ai - bi;
    } else {
      auto n = al <=> bl;
      if (n != 0) return n;
    }

    a = ar;
    b = br;
  }

  die("duplicate keys: ", in_a, " and ", in_b);
}

int main(int argc, char ** argv) try {
  if (argc != 2) die("missing filename");

  auto model_raw = jojo::read(jute::view::unsafe(argv[1]));
  jute::view model { model_raw.begin(), model_raw.size() };

  auto hdr_size = *reinterpret_cast<const uint64_t *>(model.begin());
  auto [sz, hdr, cnt] = model.subview(8, hdr_size);

  if (hdr_size != hdr.size())
    die("invalid safetensor - expecting header with size ", hdr_size, ", got ", hdr.size());
  
  auto json = jason::parse(hdr);

  namespace j = jason::ast;
  namespace jn = j::nodes;
  auto & root = j::cast<jn::dict>(json);
  hai::varray<jute::heap> sorted_keys { root.size() };
  for (auto &[k, v] : root) {
    if (*k == "__metadata__") continue;
    sorted_keys.push_back(k);
  }

  for (auto i = 0; i < sorted_keys.size(); i++) {
    for (auto j = i + 1; j < sorted_keys.size(); j++) {
      auto & ii = sorted_keys[i];
      auto & jj = sorted_keys[j];
      if (key_cmp(*ii, *jj) <= 0) continue;

      jute::heap kk = ii;
      ii = jj;
      jj = kk;
    }
  }

  for (auto key : sorted_keys) {
    putln(key, ' ');
  }
  putln();
} catch (...) {
  return 1;
}
