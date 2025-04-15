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

  for (auto i = 0; i < all.size() - 1; i++) {
    auto key = all.subview(i, 2).middle;
    auto & idx = idxs[key];
    if (idx == 0) {
      items.push_back({ key });
      idx = items.size();
    }
    items.seek(idx - 1).count++;
  }

  for (auto &[k, c]: items) {
    if (k == "") continue;
    putln(k, " = ", c);
  }
}
