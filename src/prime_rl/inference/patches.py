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
    from vllm.entrypoints.openai.models.serving import (
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


# Monkeypatch LRUCacheWorkerLoRAManager to allow loading adapter inplace without doing it every request
def monkey_patch_LRUCacheWorkerLoRAManager():
    from vllm.lora.worker_manager import LoRARequest, LRUCacheLoRAModelManager, LRUCacheWorkerLoRAManager

    # The dunder is intended. It's a private method that we're patching.
    def _patched__apply_adapters(self: LRUCacheWorkerLoRAManager, lora_requests: set[LoRARequest]) -> None:
        loras_map = {lora_request.lora_int_id: lora_request for lora_request in lora_requests if lora_request}
        if len(loras_map) > self._adapter_manager.lora_slots:
            raise RuntimeError(
                f"Number of requested LoRAs ({len(loras_map)}) is greater "
                "than the number of GPU LoRA slots "
                f"({self._adapter_manager.lora_slots})."
            )
        for lora in loras_map.values():
            ## START PATCHED CODE
            self.add_adapter(lora, force_load=False)
            ## END PATCHED CODE

    def _patched_add_adapter(
        self: LRUCacheWorkerLoRAManager, lora_request: LoRARequest, force_load: bool = True
    ) -> bool:
        # Note that this method is not thread-safe. It may be invoked multiple
        # times for the same adapter when using multiple API servers.
        # This is ok because it's currently only called from
        # the single-threaded core engine loop.

        ## START PATCHED CODE
        if lora_request.lora_int_id not in self.list_adapters() or force_load:
            ## END PATCHED CODE
            # Load the new adapter first to ensure it is actually valid, before
            # evicting any existing adapters.
            # This may cause the # of loaded lora adapters to very temporarily
            # exceed `--max-cpu-loras`.
            lora = self._load_adapter(lora_request)
            ## START PATCHED CODE
            self._adapter_manager.remove_adapter(lora.id)
            ## END PATCHED CODE

            # Loading succeeded, now check if we will exceed cache capacity and
            # evict if the oldest adapter if so
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                assert isinstance(self._adapter_manager, LRUCacheLoRAModelManager)
                self._adapter_manager.remove_oldest_adapter()
            # Then add the new adapter to the cache
            loaded = self._adapter_manager.add_adapter(lora)
        else:
            # If the lora is already loaded, just touch it to
            # update its position in the caches
            loaded = self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
        self._adapter_manager.activate_adapter(lora_request.lora_int_id)
        return loaded

    LRUCacheWorkerLoRAManager._apply_adapters = _patched__apply_adapters
    LRUCacheWorkerLoRAManager.add_adapter = _patched_add_adapter
