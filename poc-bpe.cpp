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
struct max_compression_reached {};

using dict = hai::chain<pair>;
using tk_str = hai::chain<unsigned>;

static auto create_initial_tokens() {
  dict tokens { 102400 };
  for (auto i = 0U; i < 256; i++) tokens.push_back({ i, 0 });
  return tokens;
}

static auto convert_to_pair_indices(jute::view str) {
  tk_str pairs { 102400 };
  for (auto c : str) pairs.push_back(c);
  return pairs;
}

static auto find_next_pair(const tk_str & str) {
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
  if (max_count == 1) throw max_compression_reached {};
  return items.seek(max_id - 1).key;
}

static auto compress(const tk_str & old, pair p, unsigned idx) {
  tk_str res { old.size() };
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

static auto run_one_compress(const tk_str & str, dict & d) {
  auto pair = find_next_pair(str);
  d.push_back(pair);
  return compress(str, pair, d.size() - 1);
}

static auto run_compression(jute::view in, dict & d) {
  auto str = convert_to_pair_indices(in);
  while (true) try {
    str = run_one_compress(str, d);
  } catch (max_compression_reached) {
    return str;
  }
}

int main() {
  //auto cstr = jojo::read_cstr("dom-casmurro.txt");
  //jute::view all { cstr };
  jute::view all { "o rato roeu a roupa do rei de roma" };

  auto tokens = create_initial_tokens();
  auto str = run_compression(all, tokens);

  for (auto c : str) {
    if (c < 256) put((char) c);
    else put("[", c, "]");
  }
  putln();
}
