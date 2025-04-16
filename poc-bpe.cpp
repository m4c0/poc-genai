#pragma leco tool
import hai;
import hashley;
import jojo;
import jute;
import print;

static auto create_initial_tokens() {
  hai::chain<jute::heap> tokens { 102400 };
  for (auto i = 0; i < 256; i++) {
    char buf = i;
    tokens.push_back(jute::view { &buf, 1 });
  }
  return tokens;
}

static auto find_next_pair(jute::view all) {
  struct item {
    jute::view key;
    unsigned count = 0;
  };

  hashley::niamh idxs { 1023 };
  hai::chain<item> items { 10240 };
  items.push_back({});

  unsigned max_id {};
  unsigned max_count {};
  for (auto i = 0; i < all.size() - 1; i++) {
    auto key = all.subview(i, 2).middle;
    auto & idx = idxs[key];
    if (idx == 0) {
      items.push_back({ key });
      idx = items.size();
    }
    auto c = ++items.seek(idx - 1).count;
    if (c > max_count) {
      max_id = idx;
      max_count = c;
    }
  }
  return items.seek(max_id - 1).key;
}

int main() {
  //auto cstr = jojo::read_cstr("dom-casmurro.txt");
  //jute::view all { cstr };
  jute::view all { "o rato roeu a roupa do rei de roma" };

  auto tokens = create_initial_tokens();
  auto key = find_next_pair(all);
}
