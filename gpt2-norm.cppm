export module gpt2:norm;
import :norm1;
import :reduce1k;

namespace gpt2::stages {
  export class norm {
    reduce_sum<n_ctx, n_embed> m_0;
    norm1 m_1; // sum((x - mean) ^ 2)

  public:
    norm(vee::physical_device pd, vee::buffer::type in)
      : m_0 { pd, in }
      , m_1 { pd, in, m_0.buffer() } {}

    void cmd_dispatch(vee::command_buffer cb) {
      m_0.cmd_dispatch(cb);
      m_1.cmd_dispatch(cb);
    }

    auto memory() const { return m_0.memory(); }
  };
}
