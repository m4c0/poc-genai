export module gpt2:smax;
import :smax0;
import :smax1;
import :smax2;

namespace gpt2::stages {
  export class smax {
    smax0 m_0; // max(x)
    smax1 m_1; // x = exp(x - max)
    smax2 m_2; // sum(x)

  public:
    smax(vee::physical_device pd, vee::buffer::type in)
      : m_0 { pd, in }
      , m_1 { pd, in, m_0.buffer() }
      , m_2 { pd, in } {}

    void cmd_dispatch(vee::command_buffer cb) {
      m_0.cmd_dispatch(cb);
      m_1.cmd_dispatch(cb);
      m_2.cmd_dispatch(cb);
    }
  };
}
