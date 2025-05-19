#pragma leco tool
import hai;
import hashley;
import jojo;
import jute;
import print;
import traits;

using namespace traits::ints;

class pairs {
  struct pair {
    unsigned a;
    unsigned b;
    unsigned count;
  };

  hashley::aoife<unsigned long, unsigned> m_idx { 1023 };
  hai::chain<pair> m_data { 102400 };

  auto count_of(unsigned idx) const { return m_data.seek(idx).count; }
  void print(unsigned idx) const {
    auto [a, b, c] = m_data.seek(idx);
    if (b == 0 && 32 <= a && a <= 127) putf("%c", a);
    else if (b == 0) putf("\\x%02X", a);
    else { print(a); print(b); }
  }

public:
  unsigned ping(unsigned a, unsigned b) {
    uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
    auto & idx = m_idx[key];
    if (idx) {
      m_data.seek(idx - 1).count++;
    } else {
      m_data.push_back({ a, b, 1 });
      idx = m_data.size();
    }
    return idx - 1;
  }
  unsigned ping(char c) { return ping(static_cast<unsigned>(c) & 0xFF, 0); }

  auto begin() const { return m_data.begin(); }
  auto end() const { return m_data.end(); }
  auto size() const { return m_data.size(); }

  void dump(int min_count) const {
    for (auto i = 0; i < size(); i++) {
      if (count_of(i) < min_count) continue;
      putf("%10d [", count_of(i));
      print(i);
      putln("]");
    }
    putfn(">>>>>>>>>>>>>> %d pairs", size());
  }
};

int main() {
  auto cstr = jojo::read_cstr("dom-casmurro.txt");
  jute::view all { cstr };

  pairs ps {};

#if 1
  auto a = ps.ping(all[0]);
  for (auto i = 1; i < all.size(); i++) {
    auto b = ps.ping(all[i]);
    ps.ping(a, b);
    a = b;
  }
#else
  auto a = ps.ping(all[0]);
  auto b = ps.ping(all[1]);
  auto ab = ps.ping(a, b);
  for (auto i = 2; i < all.size(); i++) {
    auto c = ps.ping(all[i]);
    auto bc = ps.ping(b, c);
    ps.ping(a, bc);
    ps.ping(ab, c);
    a = b;
    b = c;
    ab = bc;
  }
#endif

  ps.dump(100);
}
