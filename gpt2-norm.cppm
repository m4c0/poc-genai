#pragma leco add_shader "gpt2-norm1.comp"
#pragma leco add_shader "gpt2-norm2.comp"
export module gpt2:norm;
import :consts;
import :kernel;
import :reduce1k;
import :utils;
import vee;

namespace gpt2::stages {
  export class norm {
    utils::buffer m_var;
    utils::buffer m_out;
    reduce_sum<n_ctx, n_embed> m_0;
    kernel m_1; // sum((x - mean) ^ 2)
    kernel m_2;
    kernel m_3; // out

  public:
    norm(vee::physical_device pd,
         vee::buffer::type in,
         vee::buffer::type w,
         vee::buffer::type b)
      : m_var { pd, n_ctx * 32 }
      , m_out { pd, n_ctx * n_embed }
      , m_0 { pd, in }
      , m_1 { pd, "gpt2-norm1.comp.spv", in, m_0.buffer(), *m_var }
      , m_2 { pd, "gpt2-reduce1k-sum.comp.spv", *m_var, *m_var }
      , m_3 { pd, "gpt2-norm2.comp.spv",
              in, m_0.buffer(), *m_var, w, b, *m_out } {}

    void cmd_dispatch(vee::command_buffer cb, unsigned tks) {
      m_0.cmd_dispatch(cb);
      m_1.cmd_dispatch(cb, tks, 1, 24);
      m_2.cmd_dispatch(cb, tks, 1, 1);
      m_3.cmd_dispatch(cb, tks, 1, n_embed);
    }

    auto buffer() const { return *m_out; }
    auto memory() const { return m_out.memory(); }
  };
}
