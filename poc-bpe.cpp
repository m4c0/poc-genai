#pragma leco tool
#define DUMP_TEST 0
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
using tk_str = hai::varray<unsigned>;

#if DUMP_TEST
static void dump_token(const dict & d, unsigned c) {
  if (c < 256) put((char) c);
  else {
    auto [a, b] = d.seek(c);
    dump_token(d, a);
    dump_token(d, b);
  }
}
static void dump(const tk_str & str, const dict & d) {
  for (auto c : str) dump_token(d, c);
  putln();
}
#endif

static auto create_initial_tokens() {
  dict tokens { 102400 };
  for (auto i = 0U; i < 256; i++) tokens.push_back({ i, 0 });
  return tokens;
}

static auto convert_to_pair_indices(jute::view str) {
  tk_str pairs { static_cast<unsigned>(str.size()) };
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
    pair key { str[i], str[i + 1] };
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

static void compress(tk_str & old, pair p, unsigned idx) {
  auto o = 0;
  for (auto i = 0; i < old.size(); i++, o++) {
    auto a = old[i];
    if (i < old.size() - 1 && a == p.a && old[i + 1] == p.b) {
      old[o] = idx;
      i++;
    } else {
      old[o] = old[i];
    }
  }
  old.truncate(o);
}

static void run_one_compress(tk_str & str, dict & d) {
  auto pair = find_next_pair(str);
  d.push_back(pair);
  compress(str, pair, d.size() - 1);
}

static auto run_compression(jute::view in, dict & d) {
  auto str = convert_to_pair_indices(in);
  while (true) try {
    run_one_compress(str, d);
    putln(str.size(), "\t\t", d.size());
  } catch (max_compression_reached) {
    return str;
  }
}

int main() {
#if 1
  auto cstr = jojo::read_cstr("dom-casmurro.txt");
  jute::view all { cstr };
#else
  jute::view all { "o rato roeu a roupa do rei de roma" };
#endif

  auto tokens = create_initial_tokens();
  auto str = run_compression(all, tokens);
#if DUMP_TEST
  dump(str, tokens);
#endif
}
