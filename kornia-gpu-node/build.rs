fn main() {
    prost_build::Config::new()
        .extern_path(".bubbaloop.header.v1", "::bubbaloop_schemas::header::v1")
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .file_descriptor_set_path(
            std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
                .join("descriptor.bin"),
        )
        .compile_protos(&["protos/camera.proto"], &["protos/"])
        .expect("Failed to compile camera.proto");
}
