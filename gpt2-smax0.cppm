#pragma leco add_shader "gpt2-smax0.comp"
export module gpt2:smax0;
import :consts;
import :utils;
import vee;

namespace gpt2::stages {
  export class smax0 {
    vee::descriptor_pool m_dpool;
    vee::descriptor_set m_ds0;
    vee::descriptor_set m_ds1;
    vee::pipeline_layout m_pl;
    vee::c_pipeline m_p;
    utils::buffer m_out;

  public:
    smax0(vee::physical_device pd, vee::buffer::type in)
      : m_out { pd, n_head * n_ctx * 32 } {
      auto dsl = vee::create_descriptor_set_layout({
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
      });
      m_dpool = vee::create_descriptor_pool(2, { vee::storage_buffer(4) });
      m_pl = vee::create_pipeline_layout({ *dsl });

      m_p = utils::create_pipeline("gpt2-smax0.comp.spv", *m_pl);
      m_ds0 = utils::allocate_dset(*m_dpool, *dsl, in, *m_out);
      m_ds1 = utils::allocate_dset(*m_dpool, *dsl, *m_out, *m_out);
    };

    void cmd_dispatch(vee::command_buffer cb) {
      vee::cmd_bind_c_pipeline(cb, *m_p);
      vee::cmd_bind_c_descriptor_set(cb, *m_pl, 0, m_ds0);
      vee::cmd_dispatch(cb, n_head * n_ctx, 1, n_ctx);
      vee::cmd_pipeline_barrier(cb, *m_out, vee::from_compute_to_compute);
      vee::cmd_bind_c_descriptor_set(cb, *m_pl, 0, m_ds1);
      vee::cmd_dispatch(cb, n_head * n_ctx, 1, 32);
      vee::cmd_pipeline_barrier(cb, *m_out, vee::from_compute_to_compute);
    }

    auto buffer() const { return *m_out; }
  };
}
