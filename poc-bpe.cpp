#pragma leco tool
import hai;
import hashley;
import jojo;
import jute;
import print;

struct pair {
  unsigned a;
  unsigned b;
};

static auto create_initial_tokens() {
  hai::chain<pair> tokens { 102400 };
  for (auto i = 0U; i < 256; i++) tokens.push_back({ i, 0 });
  return tokens;
}

static auto convert_to_pair_indices(jute::view str) {
  hai::chain<unsigned> pairs { 102400 };
  for (auto c : str) pairs.push_back(c);
  return pairs;
}

static auto find_next_pair(const hai::chain<unsigned> & str) {
  struct item {
    pair key;
    unsigned count = 0;
  };

  hashley::siobhan idxs { 1023 };
  hai::chain<item> items { 10240 };
  items.push_back({});

  unsigned max_id {};
  unsigned max_count {};
  for (auto i = 0; i < str.size() - 1; i++) {
    pair key { str.seek(i), str.seek(i + 1) };
    auto & idx = idxs[(key.a << 16) | key.b];
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

static auto compress(const hai::chain<unsigned> & old, pair p, unsigned idx) {
  hai::chain<unsigned> res { old.size() };
  for (auto i = 0; i < old.size(); i++) {
    auto a = old.seek(i);
    if (a == p.a && i < old.size() - 1 && old.seek(i + 1) == p.b) {
      res.push_back(idx);
      i++;
    } else {
      res.push_back(a);
    }
  }
  return res;
}

int main() {
  //auto cstr = jojo::read_cstr("dom-casmurro.txt");
  //jute::view all { cstr };
  jute::view all { "o rato roeu a roupa do rei de roma" };

  auto tokens = create_initial_tokens();
  auto str = convert_to_pair_indices(all);

  auto pair = find_next_pair(str);
  tokens.push_back(pair);
  str = compress(str, pair, tokens.size() - 1);
  for (auto c : str) {
    if (c < 256) put((char) c);
    else put("[", c, "]");
  }
  putln();
}
