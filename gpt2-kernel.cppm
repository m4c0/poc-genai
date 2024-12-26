export module gpt2:kernel;
import :utils;
import jute;
import vee;

namespace gpt2 {
  class kernel {
    vee::descriptor_pool m_dpool;
    vee::descriptor_set m_ds;
    vee::pipeline_layout m_pl;
    vee::c_pipeline m_p;
    vee::buffer::type m_in;

    static auto storage(auto) { return vee::dsl_compute_storage(); }

  public:
    kernel() = default;
    kernel(vee::physical_device pd, jute::view shd, auto... bufs) {
      auto dsl = vee::create_descriptor_set_layout({ storage(bufs)...  });

      m_dpool = vee::create_descriptor_pool(1, { vee::storage_buffer(sizeof...(bufs)) });
      m_pl = vee::create_pipeline_layout({ *dsl });

      m_p = utils::create_pipeline(shd, *m_pl);
      m_ds = utils::allocate_dset(*m_dpool, *dsl, bufs...);

      ((m_in = bufs), ...);
    }

    void cmd_dispatch(vee::command_buffer cb, unsigned x, unsigned y, unsigned z) {
      vee::cmd_bind_c_pipeline(cb, *m_p);
      vee::cmd_bind_c_descriptor_set(cb, *m_pl, 0, m_ds);
      vee::cmd_dispatch(cb, x, y, z);
      vee::cmd_pipeline_barrier(cb, m_in, vee::from_compute_to_compute);
    }
  };
}
