package store;

import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.indices.CreateIndexRequest;
import org.elasticsearch.client.indices.CreateIndexResponse;
import org.elasticsearch.client.indices.GetIndexRequest;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.action.bulk.BulkProcessor;
import org.elasticsearch.action.bulk.BulkRequest;
import org.elasticsearch.action.bulk.BulkResponse;
import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.common.unit.ByteSizeUnit;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.core.TimeValue;

import java.io.IOException;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import core.WorkerTaskResult;
import core.config.ProfilerConfig;

public class NativeElasticStore implements Store {

    final private Logger LOG = LoggerFactory.getLogger(NativeElasticStore.class.getName());

    private String serverUrl;
    private String storeServer;
    private int storePort;

    private RestHighLevelClient client;
    private BulkProcessor bulkProcessor;

    public NativeElasticStore(ProfilerConfig pc) {
        String storeServer = pc.getString(ProfilerConfig.STORE_SERVER);
        int storePort = pc.getInt(ProfilerConfig.STORE_HTTP_PORT);
        this.storeServer = storeServer;
        this.storePort = storePort;
        this.serverUrl = "http://" + storeServer + ":" + storePort;
    }

    @Override
    public void initStore() {
        // Create REST client
        client = new RestHighLevelClient(
            RestClient.builder(
                new HttpHost(storeServer, storePort, "http")));

        // Create bulk processor
        BulkProcessor.Listener listener = new BulkProcessor.Listener() {
            @Override
            public void beforeBulk(long executionId, BulkRequest request) {
                // Implementation remains the same
            }

            @Override
            public void afterBulk(long executionId, BulkRequest request, BulkResponse response) {
                // Implementation remains the same
            }

            @Override
            public void afterBulk(long executionId, BulkRequest request, Throwable failure) {
                // Implementation remains the same
            }
        };

        bulkProcessor = BulkProcessor.builder(
            (request, bulkListener) -> client.bulkAsync(request, RequestOptions.DEFAULT, bulkListener),
            listener)
            .setBulkActions(-1)
            .setBulkSize(new ByteSizeValue(50, ByteSizeUnit.MB))
            .setFlushInterval(TimeValue.timeValueSeconds(5))
            .setConcurrentRequests(1)
            .build();

        // Create indices if they don't exist
        try {
            if (!client.indices().exists(new GetIndexRequest("text"), RequestOptions.DEFAULT)) {
                CreateIndexRequest createTextIndex = new CreateIndexRequest("text");
                createTextIndex.settings(Settings.builder()
                    .put("index.number_of_shards", 1)
                    .put("index.number_of_replicas", 0)
                );
                createTextIndex.mapping(getTextMapping(), XContentType.JSON);
                CreateIndexResponse createTextResponse = client.indices().create(createTextIndex, RequestOptions.DEFAULT);
                LOG.info("Created text index: " + createTextResponse.isAcknowledged());
            }

            if (!client.indices().exists(new GetIndexRequest("profile"), RequestOptions.DEFAULT)) {
                CreateIndexRequest createProfileIndex = new CreateIndexRequest("profile");
                createProfileIndex.settings(Settings.builder()
                    .put("index.number_of_shards", 1)
                    .put("index.number_of_replicas", 0)
                    .loadFromSource(getAnalysisSettings(), XContentType.JSON)
                );
                createProfileIndex.mapping(getProfileMapping(), XContentType.JSON);
                CreateIndexResponse createProfileResponse = client.indices().create(createProfileIndex, RequestOptions.DEFAULT);
                LOG.info("Created profile index: " + createProfileResponse.isAcknowledged());
            }
        } catch (IOException e) {
            LOG.error("Error creating indices", e);
        }
    }

    private String getTextMapping() {
        return "{ \"properties\": { " +
               "\"id\": { \"type\": \"long\", \"store\": true, \"index\": true }, " +
               "\"dbName\": { \"type\": \"keyword\", \"index\": false }, " +
               "\"path\": { \"type\": \"keyword\", \"index\": false }, " +
               "\"sourceName\": { \"type\": \"keyword\", \"index\": false }, " +
               "\"columnName\": { \"type\": \"keyword\", \"index\": false, \"ignore_above\": 512 }, " +
               "\"columnNameSuggest\": { \"type\": \"completion\" }, " +
               "\"text\": { \"type\": \"text\", \"store\": false, \"index\": true, \"analyzer\": \"english\", \"term_vector\": \"yes\" } " +
               "} }";
    }
    private String getProfileMapping() {
    return "{ \"properties\": { " +
           "\"id\": { \"type\": \"long\", \"index\": true }, " +
           "\"dbName\": { \"type\": \"keyword\", \"index\": false }, " +
           "\"path\": { \"type\": \"keyword\", \"index\": false }, " +
           "\"sourceNameNA\": { \"type\": \"keyword\", \"index\": true }, " +
           "\"sourceName\": { \"type\": \"text\", \"index\": true, \"analyzer\": \"aurum_analyzer\" }, " +
           "\"columnNameNA\": { \"type\": \"keyword\", \"index\": true }, " +
           "\"columnName\": { \"type\": \"text\", \"index\": true, \"analyzer\": \"aurum_analyzer\" }, " +
           "\"dataType\": { \"type\": \"keyword\", \"index\": true }, " +
           "\"totalValues\": { \"type\": \"long\", \"index\": false }, " +
           "\"uniqueValues\": { \"type\": \"long\", \"index\": false }, " +
           "\"entities\": { \"type\": \"keyword\", \"index\": true }, " +
           "\"minhash\": { \"type\": \"long\", \"index\": false }, " +
           "\"minValue\": { \"type\": \"double\", \"index\": false }, " +
           "\"maxValue\": { \"type\": \"double\", \"index\": false }, " +
           "\"avgValue\": { \"type\": \"double\", \"index\": false }, " +
           "\"median\": { \"type\": \"long\", \"index\": false }, " +
           "\"iqr\": { \"type\": \"long\", \"index\": false } " +
           "} }";
    }
    private String getAnalysisSettings() {
    return "{ \"analysis\": { " +
           "\"char_filter\": { " +
           "  \"aurum_char_filter\": { " +
           "    \"type\": \"mapping\", " +
           "    \"mappings\": [\"_=>-\", \".csv=> \"] " +
           "  } " +
           "}, " +
           "\"filter\": { " +
           "  \"english_stop\": { " +
           "    \"type\": \"stop\", " +
           "    \"stopwords\": \"_english_\" " +
           "  }, " +
           "  \"english_stemmer\": { " +
           "    \"type\": \"stemmer\", " +
           "    \"language\": \"english\" " +
           "  }, " +
           "  \"english_possessive_stemmer\": { " +
           "    \"type\": \"stemmer\", " +
           "    \"language\": \"possessive_english\" " +
           "  } " +
           "}, " +
           "\"analyzer\": { " +
           "  \"aurum_analyzer\": { " +
           "    \"tokenizer\": \"standard\", " +
           "    \"char_filter\": [\"aurum_char_filter\"], " +
           "    \"filter\": [\"english_possessive_stemmer\", \"lowercase\", \"english_stop\", \"english_stemmer\"] " +
           "  } " +
           "} " +
           "} }";
    }
    //private String getTextMapping() {
    //    // Convert your existing text_mapping to a JSON string
    //    return "{ \"properties\": { ... } }";
    //}

    //private String getProfileMapping() {
    //    // Convert your existing profile_mapping to a JSON string
    //    return "{ \"properties\": { ... } }";
    //}

    //private String getAnalysisSettings() {
    //    // Convert your existing analysis settings to a JSON string
    //    return "{ \"analysis\": { ... } }";
    //}

    @Override
    public boolean indexData(long id, String dbName, String path, String sourceName, String columnName, List<String> values) {
        IndexRequest request = new IndexRequest("text")
            .id(String.valueOf(id))
            .source("id", id,
                    "dbName", dbName,
                    "path", path,
                    "sourceName", sourceName,
                    "columnName", columnName,
                    "columnNameSuggest", columnName,
                    "text", values);

        bulkProcessor.add(request);
        return true;
    }

    @Override
    public boolean storeDocument(WorkerTaskResult wtr) {
        IndexRequest request = new IndexRequest("profile")
            .id(String.valueOf(wtr.getId()))
            .source("id", wtr.getId(),
                    "dbName", wtr.getDBName(),
                    "path", wtr.getPath(),
                    "sourceName", wtr.getSourceName(),
                    "columnNameNA", wtr.getColumnName(),
                    "columnName", wtr.getColumnName(),
                    "dataType", wtr.getDataType(),
                    "totalValues", wtr.getTotalValues(),
                    "uniqueValues", wtr.getUniqueValues(),
                    "entities", wtr.getEntities().toString(),
                    "minhash", wtr.getMH(),
                    "minValue", wtr.getMinValue(),
                    "maxValue", wtr.getMaxValue(),
                    "avgValue", wtr.getAvgValue(),
                    "median", wtr.getMedian(),
                    "iqr", wtr.getIQR());

        bulkProcessor.add(request);
        return true;
    }

    @Override
    public void tearDownStore() {
        try {
            bulkProcessor.close();
            client.close();
        } catch (IOException e) {
            LOG.error("Error closing Elasticsearch client", e);
        }
    }
}
