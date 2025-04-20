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

using tk_str = hai::varray<unsigned>;

class dict {
  hai::varray<pair> m_data { 102400 };

public:
  dict() {
    for (auto i = 0U; i < 256; i++) m_data.push_back(pair { i, 0 });
  }

  [[nodiscard]] constexpr auto operator[](unsigned idx) const { return m_data[idx]; }

  [[nodiscard]] auto push_back(pair p) {
    m_data.push_back(p);
    if (m_data.size() == m_data.capacity()) throw 1;
    return m_data.size() - 1;
  }

  [[nodiscard]] constexpr auto count() const { return m_data.size(); }

  void uncompress(unsigned c, FILE * f) {
    if (c < 256) fputc((char) c, f);
    else {
      auto [a, b] = m_data[c];
      uncompress(a, f);
      uncompress(b, f);
    }
  }
  void uncompress(const tk_str & str, const char * filename) {
    FILE * f = fopen(filename, "wb");
    for (auto c : str) uncompress(c, f);
    fclose(f);
  }

  void dump_table() {
    for (auto i = 256; i < m_data.size(); i++) {
      uncompress(i, stdout);
      putln();
    }
  }

  void read(const char * filename) {
    FILE * f = fopen(filename, "rb");
    if (!f) throw 2;

    if (0 != fseek(f, 0, SEEK_END)) throw 3;
    if (ftell(f) <= 0) throw 4;

    unsigned count = ftell(f) / sizeof(pair);
    m_data.set_capacity(count);
    m_data.expand(count);
    fseek(f, 0, SEEK_SET);
    if (fread(m_data.begin(), sizeof(pair), count, f) != count) throw 5;
    fclose(f);
  }
  void write(const char * filename) {
    FILE * f = fopen(filename, "wb");
    fwrite(m_data.begin(), sizeof(pair), m_data.size(), f);
    fclose(f);
  }
};

static auto convert_to_pair_indices(jute::view str) {
  tk_str pairs { static_cast<unsigned>(str.size()) };
  for (unsigned c : str) pairs.push_back(c & 0xFF);
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
      if (m_i_count == 102400) throw 6;
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

int main() try {
  const char * in = "dom-casmurro.txt";

  auto cstr = jojo::read_cstr(jute::view::unsafe(in));
  jute::view all { cstr };

  dict tokens {};

  tokens.read("out/dump.bpe");

  // auto str = run_compression(all, tokens);
  // tokens.uncompress(str, "out/dump.txt");
  // tokens.write("out/dump.bpe");

  tokens.dump_table();
} catch (int e) {
  return e;
}
