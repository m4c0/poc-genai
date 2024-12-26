export module gpt2:norm0;
import :consts;
import :reduce1k;
import vee;

namespace gpt2::stages {
  export struct norm0 : reduce1k<n_ctx, n_embed> {
    norm0(vee::physical_device pd, vee::buffer::type in)
      : reduce1k { pd, in, "gpt2-reduce1k-sum.comp.spv" } {};
  };
}
