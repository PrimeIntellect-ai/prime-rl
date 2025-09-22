# Troubleshooting

- Especially for large training workloads with large batch sizes, you may find yourself getting API timeout errors because your OS limits the number of open files. If this is the case, you can increase the maximum number of open files with

  ```bash
  ulimit -n 32000
  ```
