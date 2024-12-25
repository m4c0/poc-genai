#pragma leco add_shader "gpt2-smax3.comp"
export module gpt2:smax3;
import :consts;
import :utils;
import vee;

namespace gpt2::stages {
  export class smax3 {
    vee::descriptor_pool m_dpool;
    vee::descriptor_set m_ds;
    vee::pipeline_layout m_pl;
    vee::c_pipeline m_p;
    vee::buffer::type m_in;

  public:
    smax3(vee::physical_device pd, vee::buffer::type in, vee::buffer::type mx) {
      m_in = in;

      auto dsl = vee::create_descriptor_set_layout({
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
      });
      m_dpool = vee::create_descriptor_pool(1, { vee::storage_buffer(2) });
      m_pl = vee::create_pipeline_layout({ *dsl });

      m_p = utils::create_pipeline("gpt2-smax3.comp.spv", *m_pl);
      m_ds = utils::allocate_dset(*m_dpool, *dsl, in, mx);
    };

    void cmd_dispatch(vee::command_buffer cb) {
      vee::cmd_bind_c_pipeline(cb, *m_p);
      vee::cmd_bind_c_descriptor_set(cb, *m_pl, 0, m_ds);
      vee::cmd_dispatch(cb, n_head, n_ctx, n_ctx);
      vee::cmd_pipeline_barrier(cb, m_in, vee::from_compute_to_compute);
    }
  };
}
