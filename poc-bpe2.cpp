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
  void ping(unsigned a, unsigned b) {
    uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
    auto & idx = m_idx[key];
    if (idx) {
      m_data.seek(idx - 1).count++;
      return;
    }

    m_data.push_back({ a, b, 1 });
    idx = m_data.size();
  }

  auto begin() const { return m_data.begin(); }
  auto end() const { return m_data.end(); }
  auto size() const { return m_data.size(); }
};

int main() {
  auto cstr = jojo::read_cstr("dom-casmurro.txt");
  jute::view all { cstr };

  pairs ps {};

  for (char c : all) {
    ps.ping(c & 0xFF, 0);
  }

  for (auto [a, b, c]: ps) {
    if (b != 0) putln(a, " ", b, " ", c);
    else if (32 <= a && a <= 127) putfn("%c %d %d", a, b, c);
    else putfn("x%02X %d %d", a, b, c);
  }
  putfn(">>>>>>>>>>>>>> %d pairs", ps.size());
}
