#pragma leco add_shader "gpt2-smax2.comp"
export module gpt2:smax2;
import :reduce1k;
import vee;

namespace gpt2::stages {
  export struct smax2 : reduce1k<n_head * n_ctx, n_ctx> {
    smax2(vee::physical_device pd, vee::buffer::type in)
      : reduce1k { pd, in, "gpt2-smax2.comp.spv" } {}
  };
}
