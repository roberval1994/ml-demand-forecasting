# Data / Dados

The dataset is **downloaded automatically** on first run into `data/raw/` and is not
versioned. The pipeline is idempotent: once the file exists, it is reused.

O dataset é **baixado automaticamente** na primeira execução para `data/raw/` e não é
versionado. O pipeline é idempotente: uma vez baixado, o arquivo é reutilizado.

## Source / Fonte

[UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
— hourly rentals (`hour.csv`), public, no licensing restrictions.

## Troubleshooting: SSL certificate errors / Erros de certificado SSL

If the download fails with `CERTIFICATE_VERIFY_FAILED`, there are two common causes:

Se o download falhar com `CERTIFICATE_VERIFY_FAILED`, há duas causas comuns:

1. **Antivirus/proxy HTTPS inspection** (e.g. Avast, Kaspersky, corporate proxy). The
   traffic is re-signed by a local CA stored only in the operating-system trust store.
   The pipeline handles this automatically via the [`truststore`](https://pypi.org/project/truststore/)
   package (installed through `requirements.txt`), which validates against the OS store.
2. **Expired certificate on the UCI server** (a recurring, transient issue). The pipeline
   falls back to the `certifi` CA bundle and to a mirror.

> Antivírus/proxy que inspecionam HTTPS (ex.: Avast) re-assinam o tráfego com uma CA local
> que só existe no repositório de certificados do sistema. O projeto resolve isso com o
> pacote `truststore`, que valida pela cadeia do próprio sistema operacional. A verificação
> SSL **nunca** é desabilitada.

### Manual download / Download manual

If every source fails, download the archive from the UCI link above, extract `hour.csv`,
and place it in:

Se todas as fontes falharem, baixe o arquivo do link da UCI acima, extraia o `hour.csv`
e coloque-o em:

```
data/raw/hour.csv
```

Then re-run the pipeline. / Depois rode o pipeline novamente.
