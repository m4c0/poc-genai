#pragma leco add_shader "gpt2-wtewpe.comp"
export module gpt2:wtewpe;
import :consts;
import :utils;
import vee;

namespace gpt2::stages {
  export class wtewpe {
    vee::descriptor_pool m_dpool;
    vee::descriptor_set m_ds;
    vee::pipeline_layout m_pl;
    vee::c_pipeline m_p;
    utils::buffer m_out;

  public:
    wtewpe(vee::physical_device pd, vee::buffer::type wte, vee::buffer::type wpe, vee::buffer::type tks) 
      : m_out { pd, n_ctx * n_embed } {
      auto dsl = vee::create_descriptor_set_layout({
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
      });
      m_dpool = vee::create_descriptor_pool(1, { vee::storage_buffer(4) });
      m_pl = vee::create_pipeline_layout(*dsl);

      m_p = utils::create_pipeline("gpt2-wtewpe.comp.spv", *m_pl);
      m_ds = utils::allocate_dset(*m_dpool, *dsl, wte, wpe, tks, *m_out);
    }

    void cmd_dispatch(vee::command_buffer cb, unsigned tks) {
      vee::cmd_bind_c_pipeline(cb, *m_p);
      vee::cmd_bind_c_descriptor_set(cb, *m_pl, 0, m_ds);
      vee::cmd_dispatch(cb, tks, n_embed, 1);
      vee::cmd_pipeline_barrier(cb, *m_out, vee::from_compute_to_compute);
    }

    auto buffer() const { return *m_out; }
    auto memory() const { return m_out.memory(); }
  };
};
