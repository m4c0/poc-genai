#pragma leco tool
#include <stdio.h>

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

  [[nodiscard]] constexpr auto operator[](unsigned idx) const { return m_data[idx]; }

  [[nodiscard]] auto push_back(pair p) {
    m_data.push_back(p);
    if (m_data.size() == m_data.capacity()) throw 0;
    return m_data.size() - 1;
  }

  [[nodiscard]] constexpr auto data() const { return m_data.begin(); }
  [[nodiscard]] constexpr auto count() const { return m_data.size(); }
};

using tk_str = hai::varray<unsigned>;

static void uncompress_token(FILE * f, const dict & d, unsigned c) {
  if (c < 256) fputc((char) c, f);
  else {
    auto [a, b] = d[c];
    uncompress_token(f, d, a);
    uncompress_token(f, d, b);
  }
}
static void uncompress(FILE * f, const tk_str & str, const dict & d) {
  for (auto c : str) uncompress_token(f, d, c);
}

static auto convert_to_pair_indices(jute::view str) {
  tk_str pairs { static_cast<unsigned>(str.size()) };
  for (auto c : str) pairs.push_back(c);
  return pairs;
}

struct g_item {
  pair key;
  unsigned count;
};
static g_item g_items[102400] {};
class pair_counts {
  hashley::siobhan m_idxs { 7919 };
  unsigned m_i_count { 1 };
public:
  void incr(unsigned a, unsigned b) {
    auto & idx = m_idxs[(a << 16) | b];
    if (idx == 0) {
      g_items[m_i_count++] = {{ a, b }, 0};
      if (m_i_count == 102400) throw 0;
      idx = m_i_count;
    }
    ++g_items[idx - 1].count;
  }

  [[nodiscard]] constexpr auto begin() const { return g_items; }
  [[nodiscard]] constexpr auto end() const { return g_items + m_i_count; }
};
static auto find_next_pair(const tk_str & str) {
  pair_counts counts {};

  pair max_pair {};
  unsigned max_count {};
  for (auto i = 0; i < str.size() - 1; i++) counts.incr(str[i], str[i + 1]);
  for (auto [key, count] : counts) {
    if (count <= max_count) continue;
    max_pair = key;
    max_count = count;
  }
  if (max_count == 1) throw max_compression_reached {};
  return max_pair;
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
    putln(t.millis(), "\t\t", str.size(), "\t\t", d.count());
    return str;
  }
}

int main() {
  const char * in = "lorem-ipsum.txt";

  auto cstr = jojo::read_cstr(jute::view::unsafe(in));
  jute::view all { cstr };

  dict tokens {};
  auto str = run_compression(all, tokens);

  FILE * f = fopen("out/dump.txt", "wb");
  uncompress(f, str, tokens);
  fclose(f);

  f = fopen("out/dump.bpe", "wb");
  fwrite(tokens.data(), sizeof(pair), tokens.count(), f);
  fclose(f);
}
