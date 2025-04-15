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
    vee::update_descriptor_set(ds, idx, buf);
    update_dset(ds, idx + 1, bufs...);
  }
  auto allocate_dset(auto dpool, auto dsl, auto... bufs) {
    auto ds = vee::allocate_descriptor_set(dpool, dsl);
    update_dset(ds, 0, bufs...);
    return ds;
  }
}
