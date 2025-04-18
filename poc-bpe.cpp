#pragma leco tool
#define DUMP_TEST 0
import hai;
import hashley;
import jojo;
import jute;
import print;
import sitime;

struct pair {
  unsigned a;
  unsigned b;
};
struct max_compression_reached {};

class dict {
  hai::varray<pair> m_data { 102400 };

public:
  dict() {
    for (auto i = 0U; i < 256; i++) m_data.push_back(pair { i, 0 });
  }

  [[nodiscard]] auto push_back(pair p) {
    m_data.push_back(p);
    if (m_data.size() == 102400) throw 0;
    return m_data.size() - 1;
  }

  [[nodiscard]] auto count() const {
    return m_data.size();
  }
};

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

static auto convert_to_pair_indices(jute::view str) {
  tk_str pairs { static_cast<unsigned>(str.size()) };
  for (auto c : str) pairs.push_back(c);
  return pairs;
}

struct item {
  pair key;
  unsigned count = 0;
};
static item items[102400] {};
static auto find_next_pair(const tk_str & str) {
  hashley::siobhan idxs { 7919 };
  unsigned i_count { 1 };

  unsigned max_id {};
  unsigned max_count {};
  for (auto i = 0; i < str.size() - 1; i++) {
    pair key { str[i], str[i + 1] };
    auto & idx = idxs[(key.a << 16) | key.b];
    if (idx == 0) {
      items[i_count++] = { key };
      if (i_count == 102400) throw 0;
      idx = i_count;
    }
    auto c = ++items[idx - 1].count;
    if (c > max_count) {
      max_id = idx;
      max_count = c;
    }
  }
  if (max_count == 1) throw max_compression_reached {};
  return items[max_id - 1].key;
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
  compress(str, pair, d.push_back(pair));
}

static auto run_compression(jute::view in, dict & d) {
  auto str = convert_to_pair_indices(in);
  int count {};
  sitime::stopwatch t {};
  while (true) try {
    run_one_compress(str, d);
    if (count++ % 100 == 0) {
      putln(t.millis(), "\t\t", str.size(), "\t\t", d.count());
      t = {};
    }
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

  dict tokens {};
  auto str = run_compression(all, tokens);
#if DUMP_TEST
  dump(str, tokens);
#endif
}
