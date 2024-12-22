#pragma leco add_shader "gpu.comp"
export module gpu;
import dotz;
import vee;

struct buf_mem {
  vee::buffer buf;
  vee::device_memory mem;
};
export class gpu {
  static constexpr const auto mat_count = 4;
  using upc = dotz::ivec4;

  vee::instance m_i;
  vee::debug_utils_messenger m_dbg;
  vee::device m_d;
  vee::queue m_q;
  vee::descriptor_set_layout m_dsl;
  vee::pipeline_layout m_pl;
  vee::descriptor_pool m_dpool;
  vee::descriptor_set m_ds;
  buf_mem m_mats[mat_count];
  vee::c_pipeline m_p;
  vee::command_pool m_cp;
  vee::command_buffer m_cb;
  vee::fence m_f;

public:
  gpu(unsigned mat_mem_sz) {
    m_i = vee::create_instance("gpt-gpu");
    m_dbg = vee::create_debug_utils_messenger();
    const auto & [pd, qf] = vee::find_physical_device_with_universal_queue(nullptr);
    m_d = vee::create_single_queue_device(pd, qf);
    m_q = vee::get_queue_for_family(qf);

    m_dsl = vee::create_descriptor_set_layout({
      vee::dsl_compute_storage(),
      vee::dsl_compute_storage(),
      vee::dsl_compute_storage(),
      vee::dsl_compute_storage(),
    });
    m_pl = vee::create_pipeline_layout({ *m_dsl }, { vee::compute_push_constant_range<upc>() });

    m_dpool = vee::create_descriptor_pool(1, { vee::storage_buffer(mat_count) });

    m_ds = vee::allocate_descriptor_set(*m_dpool, *m_dsl);

    for (auto i = 0; i < mat_count; i++) {
      auto &[b, m] = m_mats[i];
      m = vee::create_host_buffer_memory(pd, mat_mem_sz);
      b = vee::create_buffer(mat_mem_sz, vee::buffer_usage::storage_buffer);
      vee::bind_buffer_memory(*b, *m, 0);
      vee::update_descriptor_set_with_storage(m_ds, i, *b);
    }

    auto kern = vee::create_shader_module_from_resource("gpu.comp.spv");
    m_p = vee::create_compute_pipeline(*m_pl, *kern, "main");

    m_cp = vee::create_command_pool(qf);
    m_cb = vee::allocate_primary_command_buffer(*m_cp);

    m_f = vee::create_fence_signaled();
  }

  void load(int idx, auto && fn) {
    auto p = static_cast<float *>(vee::map_memory(*m_mats[idx].mem));
    fn(p);
    vee::unmap_memory(*m_mats[idx].mem);
  }
  void run(int i, int j, int k) {
    dotz::ivec4 ijk { i, j, k, 0 };
    vee::wait_and_reset_fence(*m_f);
    vee::begin_cmd_buf_one_time_submit(m_cb);
    vee::cmd_bind_c_pipeline(m_cb, *m_p);
    vee::cmd_bind_c_descriptor_set(m_cb, *m_pl, 0, m_ds);
    vee::cmd_push_compute_constants(m_cb, *m_pl, &ijk);
    vee::cmd_dispatch(m_cb, i, j, 1);
    vee::end_cmd_buf(m_cb);
    vee::queue_submit({
      .queue = m_q,
      .fence = *m_f,
      .command_buffer = m_cb
    });
    vee::wait_for_fence(*m_f);
  }
};
