#pragma leco tool
import hai;
import hashley;
import jojo;
import jute;
import print;

struct item {
  jute::view key;
  unsigned count = 0;
};
int main() {
  //auto cstr = jojo::read_cstr("dom-casmurro.txt");
  //jute::view all { cstr };
  jute::view all { "o rato roeu a roupa do rei de roma" };

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

  putln(max_count, "-", items.seek(max_id - 1).key);
}
