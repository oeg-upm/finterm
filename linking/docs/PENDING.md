# Estado y problemas pendientes

## Distribución experimental

El cierre dinámico de código/recursos se incluye y sus imports y contratos se
prueban offline. El paquete base se puede instalar sin dependencias GPU. Los
wrappers históricos con imports imperativos de experimentos no se presentan
como CLIs autónomas; se reutilizan solo por sus adaptadores/proyecciones originales.

La licencia de distribución pública del código no venía indicada en los materiales
recibidos. No se ha inventado una licencia ni una titularidad. El responsable del
proyecto debe fijarla antes de una publicación pública; esto no impide las pruebas
internas autorizadas del paquete experimental. No se publicó ni se hizo push.

## Impide declarar full validado

- Ejecutar el notebook en la H100 real y revisar la evidencia de las tres cargas.
- Confirmar la combinación de dependencias instalada y conservar freeze transitivo.
- Demostrar traducción válida, gate positivo y definición válida reales, además
  de embeddings nuevos y Top 3/5; los mocks y el archivo account no satisfacen esto.
- Comprobar memoria real, cierre de procesos, recursos persistentes y warm-cache.
- Encontrar otros casos genuinos si los datos Wikidata actuales ya no requieren
  DefTerm o sus gates abstienen. No se puede garantizar de antemano una definición.

Estos puntos no se ocultan como validaciones completadas. La distribución sigue marcada experimental. Los informes de nuevas ejecuciones conservan pendientes, abstenciones y errores.

## Restricciones originales, no defectos corregidos silenciosamente

- Inglés para esta versión; spaCy/modelo de lemas requerido en recuperación online.
- Emparejamiento literal, deduplicación de contextos y límite original de diez.
- Textos unipalabra, traducciones pendientes y candidatos sin score se conservan.
- Los cambios de recetas/modelos/revisiones/límites por salida no son overrides
  libres: los contratos y las cachés exigen identidades exactas.
- La validación de estructura/EOS no verifica fidelidad semántica. Las definiciones
  siguen etiquetadas no verificadas semánticamente aunque puedan puntuarse.
- La cache no reintenta automáticamente fallos comprometidos/claims ambiguos.
- Top 1 no es selección definitiva; no hay umbral, gold completo ni certificación
  de que la entidad primera sea la correcta.

## Investigación posterior fuera de esta versión

Expansión semántica nueva de consultas, reconciliación SPARQL/API, exclusión de
publicaciones por P31, reranking/compatibilidad ontológica, evaluación de precisión
y adaptación a otros modelos/idiomas. No se han incorporado al comportamiento
original. Integración con external extractor y coordinación de sus modelos también son
trabajos posteriores; la entrada texto + lista ya permite un adaptador pequeño.
