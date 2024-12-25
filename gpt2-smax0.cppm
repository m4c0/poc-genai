#pragma leco add_shader "gpt2-smax0.comp"
export module gpt2:smax0;
import :consts;
import jute;
import vee;

namespace gpt2::stages {
  export class smax0 {
    vee::descriptor_pool m_dpool;
    vee::descriptor_set m_ds;
    vee::pipeline_layout m_pl;
    vee::c_pipeline m_p;
    vee::buffer m_out;
    vee::device_memory m_mem;

    static auto create_pipeline(jute::view shd, vee::pipeline_layout::type pl) {
      auto k = vee::create_shader_module_from_resource(shd);
      return vee::create_compute_pipeline(pl, *k, "main");
    } 
  public:
    smax0(vee::physical_device pd, vee::buffer::type in) {
      m_out = vee::create_buffer(n_head * n_ctx * sizeof(float), vee::buffer_usage::storage_buffer);
      m_mem = vee::create_host_buffer_memory(pd, *m_out);
      vee::bind_buffer_memory(*m_out, *m_mem, 0);

      auto dsl = vee::create_descriptor_set_layout({
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
      });
      m_dpool = vee::create_descriptor_pool(1, { vee::storage_buffer(2) });
      m_pl = vee::create_pipeline_layout({ *dsl });

      m_ds = vee::allocate_descriptor_set(*m_dpool, *dsl);
      m_p = create_pipeline("gpt2-smax0.comp.spv", *m_pl);
      vee::update_descriptor_set_with_storage(m_ds, 0, in);
      vee::update_descriptor_set_with_storage(m_ds, 1, *m_out);
    };

    void cmd_dispatch(vee::command_buffer cb) {
      vee::cmd_bind_c_pipeline(cb, *m_p);
      vee::cmd_bind_c_descriptor_set(cb, *m_pl, 0, m_ds);
      vee::cmd_dispatch(cb, n_head, n_ctx, n_ctx);
      vee::cmd_pipeline_barrier(cb, *m_out, vee::from_compute_to_compute);
    }

    auto memory() const { return *m_mem; }
  };
}
