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

  auto begin() const { return m_data.begin(); }
  auto end() const { return m_data.end(); }
  auto size() const { return m_data.size(); }

  auto count_of(unsigned idx) const { return m_data.seek(idx).count; }
  void put(unsigned idx) const {
    auto [a, b, c] = m_data.seek(idx);
    if (b == 0 && 32 <= a && a <= 127) putf("%c", a);
    else if (b == 0) putf("\\x%02X", a);
    else { put(a); put(b); }
  }
};

int main() {
  auto cstr = jojo::read_cstr("dom-casmurro.txt");
  jute::view all { cstr };

  pairs ps {};

  for (char c : all) {
    ps.ping(c & 0xFF, 0);
  }

  for (auto i = 0; i < ps.size(); i++) {
    ps.put(i);
    putln(" ", ps.count_of(i));
  }
  putfn(">>>>>>>>>>>>>> %d pairs", ps.size());
}
