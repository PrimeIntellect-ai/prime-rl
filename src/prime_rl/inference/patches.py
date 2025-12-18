# Monkeypatch PrometheusStatLogger to avoid NotImplementedError for LoRA in DP mode
def monkey_patch_prometheus_stat_logger_for_lora_in_dp_mode():
    from vllm.v1.metrics import loggers as vllm_metrics_loggers

    _original_prometheus_stat_logger_init = vllm_metrics_loggers.PrometheusStatLogger.__init__

    def _patched_prometheus_stat_logger_init(self, vllm_config, engine_indexes=None):
        """Patched init that temporarily disables lora_config to skip the DP mode check."""
        original_lora_config = vllm_config.lora_config
        vllm_config.lora_config = None
        try:
            _original_prometheus_stat_logger_init(self, vllm_config, engine_indexes)
        finally:
            vllm_config.lora_config = original_lora_config
        # Re-initialize LoRA metrics if needed (after the DP check is bypassed)
        if original_lora_config is not None:
            self.labelname_max_lora = "max_lora"
            self.labelname_waiting_lora_adapters = "waiting_lora_adapters"
            self.labelname_running_lora_adapters = "running_lora_adapters"
            self.max_lora = original_lora_config.max_loras
            self.gauge_lora_info = vllm_metrics_loggers.PrometheusStatLogger._gauge_cls(
                name="vllm:lora_requests_info",
                documentation="Running stats on lora requests.",
                multiprocess_mode="sum",
                labelnames=[
                    self.labelname_max_lora,
                    self.labelname_waiting_lora_adapters,
                    self.labelname_running_lora_adapters,
                ],
            )

    vllm_metrics_loggers.PrometheusStatLogger.__init__ = _patched_prometheus_stat_logger_init


# Monkeypatch LoadLoRAAdapter to allow loading the same adapter multiple times
def monkey_patch_load_lora_adapter():
    from vllm.entrypoints.openai.serving_models import (
        ErrorResponse,
        HTTPStatus,
        LoadLoRAAdapterRequest,
        LoRARequest,
        OpenAIServingModels,
        create_error_response,
        logger,
    )

    async def _patched_load_lora_adapter(
        self: OpenAIServingModels, request: LoadLoRAAdapterRequest, base_model_name: str | None = None
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            lora_path = request.lora_path
            ## START PATCHED CODE
            if lora_name in self.lora_requests:
                lora_request = self.lora_requests[lora_name]
                lora_request.lora_path = lora_path
            else:
                unique_id = self.lora_id_counter.inc(1)
                lora_request = LoRARequest(lora_name=lora_name, lora_int_id=unique_id, lora_path=lora_path)
            ## END PATCHED CODE
            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name

            # Validate that the adapter can be loaded into the engine
            # This will also preload it for incoming requests
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                error_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                if "No adapter found" in str(e):
                    error_type = "NotFoundError"
                    status_code = HTTPStatus.NOT_FOUND

                return create_error_response(message=str(e), err_type=error_type, status_code=status_code)

            self.lora_requests[lora_name] = lora_request
            logger.info("Loaded new LoRA adapter: name '%s', path '%s'", lora_name, lora_path)
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    OpenAIServingModels.load_lora_adapter = _patched_load_lora_adapter
