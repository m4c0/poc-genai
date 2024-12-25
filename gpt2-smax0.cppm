#pragma leco add_shader "gpt2-smax0.comp"
export module gpt2:smax0;
import :consts;
import :reduce1k;
import vee;

namespace gpt2::stages {
  export struct smax0 : reduce1k {
    smax0(vee::physical_device pd, vee::buffer::type in)
      : reduce1k { pd, in, "gpt2-smax0.comp" } {
    }
  };
}
