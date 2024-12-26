#pragma leco add_shader "gpt2-norm1.comp"
#pragma leco add_shader "gpt2-norm2.comp"
export module gpt2:norm;
import :kernel;
import :reduce1k;

namespace gpt2::stages {
  export class norm {
    utils::buffer m_var;
    utils::buffer m_out;
    reduce_sum<n_ctx, n_embed> m_0;
    kernel<n_ctx, 1, 24> m_1; // sum((x - mean) ^ 2)
    kernel<n_ctx, 1, 1> m_2;
    kernel<n_ctx, 1, n_embed> m_3; // out

  public:
    norm(vee::physical_device pd,
         vee::buffer::type in,
         vee::buffer::type w,
         vee::buffer::type b)
      : m_var { pd, n_ctx * n_embed * 32 }
      , m_out { pd, n_ctx * n_embed * n_embed }
      , m_0 { pd, in }
      , m_1 { pd, "gpt2-norm1.comp.spv", in, m_0.buffer(), *m_var }
      , m_2 { pd, "gpt2-reduce1k-sum.comp.spv", *m_var, *m_var }
      , m_3 { pd, "gpt2-norm2.comp.spv",
              in, m_0.buffer(), *m_var, w, b, *m_out } {}

    void cmd_dispatch(vee::command_buffer cb) {
      m_0.cmd_dispatch(cb);
      m_1.cmd_dispatch(cb);
      m_2.cmd_dispatch(cb);
      m_3.cmd_dispatch(cb);
    }

    auto memory() const { return m_var.memory(); }
  };
}
