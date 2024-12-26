#pragma leco add_shader "gpt2-norm1.comp"
export module gpt2:norm1;
import :consts;
import :utils;
import vee;

namespace gpt2::stages {
  class norm1 {
    utils::kernel<n_ctx, 24, 1> m_kern;

  public:
    norm1(vee::physical_device pd, vee::buffer::type in, vee::buffer::type mn) {
      m_kern = { pd, "gpt2-norm1.comp.spv", in, mn };
    };

    void cmd_dispatch(vee::command_buffer cb) {
      m_kern.cmd_dispatch(cb);
    }
  };
}
