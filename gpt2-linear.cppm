#pragma leco add_shader "gpt2-linear.comp"
export module gpt2:linear;
import :kernel;
import :utils;

namespace gpt2::stages {
  export template<unsigned I, unsigned J, unsigned K>
  class linear {
    vee::descriptor_pool m_dpool;
    vee::descriptor_set m_ds;
    vee::pipeline_layout m_pl;
    vee::c_pipeline m_p;
    vee::buffer::type m_in;
    utils::buffer m_out;

  public:
    linear(vee::physical_device pd,
        vee::buffer::type in,
        vee::buffer::type w,
        vee::buffer::type b)
      : m_out { pd, I * J } {
      auto dsl = vee::create_descriptor_set_layout({
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
        vee::dsl_compute_storage(),
      });
      m_dpool = vee::create_descriptor_pool(1, { vee::storage_buffer(4) });
      m_pl = vee::create_pipeline_layout({ *dsl }, { vee::compute_push_constant_range<unsigned>() });

      m_p = utils::create_pipeline("gpt2-linear.comp.spv", *m_pl);
      m_ds = utils::allocate_dset(*m_dpool, *dsl, in, w, b, *m_out);
    }

    void cmd_dispatch(vee::command_buffer cb, unsigned i = I) {
      unsigned k = K;
      vee::cmd_bind_c_pipeline(cb, *m_p);
      vee::cmd_bind_c_descriptor_set(cb, *m_pl, 0, m_ds);
      vee::cmd_push_compute_constants(cb, *m_pl, &k);
      vee::cmd_dispatch(cb, i, J, 1);
      vee::cmd_pipeline_barrier(cb, *m_out, vee::from_compute_to_compute);
    }

    auto buffer() const { return *m_out; }
    auto memory() const { return m_out.memory(); }
  };
}
