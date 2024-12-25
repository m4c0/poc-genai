#pragma leco app

#include <stdio.h>

import gpt2;
import jason;
import jojo;
import jute;
import print;
import traits;
import vee;
import yoyo;

using namespace gpt2;

using namespace traits::ints;
namespace j = jason::ast;
namespace jn = j::nodes;

static auto create_pipeline(jute::view shd, auto pl) {
  auto k_0 = vee::create_shader_module_from_resource(shd);
  return vee::create_compute_pipeline(pl, *k_0, "main");
}

int main() try {
  auto i = vee::create_instance("test");
  auto dbg = vee::create_debug_utils_messenger();
  const auto & [pd, qf] = vee::find_physical_device_with_universal_queue(nullptr);
  auto d = vee::create_single_queue_device(pd, qf);
  auto q = vee::get_queue_for_family(qf);

  constexpr const auto n = n_head * n_ctx * n_ctx;
  auto buf = vee::create_buffer(n * sizeof(float), vee::buffer_usage::storage_buffer);
  auto mem = vee::create_host_buffer_memory(pd, *buf);
  vee::bind_buffer_memory(*buf, *mem, 0);

  stages::smax0 smax0 { pd, *buf };

  auto cpool = vee::create_command_pool(qf);
  auto cb = vee::allocate_primary_command_buffer(*cpool);

  auto * ptr = static_cast<float *>(vee::map_memory(*mem));
  for (auto i = 0; i < n; i++) {
    ptr[i] = i + 1;
  }
  vee::unmap_memory(*mem);

  vee::begin_cmd_buf_one_time_submit(cb);
  smax0.cmd_dispatch(cb);
  vee::end_cmd_buf(cb);

  auto f = vee::create_fence_signaled();
  vee::wait_and_reset_fence(*f);
  vee::queue_submit({
    .queue = q,
    .fence = *f,
    .command_buffer = cb,
  });
  vee::wait_for_fence(*f);

  auto out = static_cast<float *>(vee::map_memory(smax0.memory()));
  for (auto h = 0; h < n_head; h++) {
    if (h > 2 && h < n_head - 2) continue;
    for (auto i = 0; i < n_ctx; i++) {
      if (i > 2 && i < n_ctx - 2) continue;

      auto hi = h * n_ctx * n_ctx + i * n_ctx;
      for (auto j = 0; j < n_ctx; j++) {
        if (j > 8 && j < n_ctx - 8) continue;
        auto n = static_cast<int>(out[hi + j]);
        putf("%8d ", n);
      }
      putln();
    }
  }
  vee::unmap_memory(smax0.memory());
} catch (...) {
  return 1;
}
