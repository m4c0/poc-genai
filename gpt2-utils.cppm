export module gpt2:utils;
import jute;
import vee;

namespace gpt2::utils {
  export class buffer {
    vee::buffer m_buf;
    vee::device_memory m_mem;

  public:
    explicit buffer(vee::physical_device pd, unsigned n) {
      m_buf = vee::create_buffer(n * sizeof(float), vee::buffer_usage::storage_buffer);
      m_mem = vee::create_host_buffer_memory(pd, *m_buf);
      vee::bind_buffer_memory(*m_buf, *m_mem, 0);
    }

    auto operator*() const { return *m_buf; }
    auto memory() const { return *m_mem; }
  };

  auto create_pipeline(jute::view shd, vee::pipeline_layout::type pl) {
    auto k = vee::create_shader_module_from_resource(shd);
    return vee::create_compute_pipeline(pl, *k, "main");
  } 

  void update_dset(auto ds, unsigned idx) {}
  void update_dset(auto ds, unsigned idx, auto buf, auto... bufs) {
    vee::update_descriptor_set_with_storage(ds, idx, buf);
    update_dset(ds, idx + 1, bufs...);
  }
  auto allocate_dset(auto dpool, auto dsl, auto... bufs) {
    auto ds = vee::allocate_descriptor_set(dpool, dsl);
    update_dset(ds, 0, bufs...);
    return ds;
  }

  template<unsigned X, unsigned Y, unsigned Z>
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

    void cmd_dispatch(vee::command_buffer cb) {
      vee::cmd_bind_c_pipeline(cb, *m_p);
      vee::cmd_bind_c_descriptor_set(cb, *m_pl, 0, m_ds);
      vee::cmd_dispatch(cb, X, Y, Z);
      vee::cmd_pipeline_barrier(cb, m_in, vee::from_compute_to_compute);
    }
  };
}
