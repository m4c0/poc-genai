export module gpt2:norm;
import :reduce1k;

namespace gpt2::stages {
  export class norm {
    reduce_sum<n_ctx, n_embed> m_0;

  public:
    norm(vee::physical_device pd, vee::buffer::type in)
      : m_0 { pd, in } {}

    void cmd_dispatch(vee::command_buffer cb) {
      m_0.cmd_dispatch(cb);
    }

    auto memory() const { return m_0.memory(); }
  };
}
