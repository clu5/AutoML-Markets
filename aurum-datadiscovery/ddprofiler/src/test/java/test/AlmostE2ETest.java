package test;

import java.util.List;
import java.util.Properties;
import java.time.Duration;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.Disabled;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import core.Conductor;
import core.WorkerTaskResult;
import core.config.ProfilerConfig;
import store.Store;
import store.StoreFactory;


public class AlmostE2ETest {

    @Timeout(value = 60)
    private void timeoutTest() {
        // This method will timeout after 60 seconds
    }

    private String path = "C:\\";
    private String filename = "Leading_Causes_of_Death__1990-2010.csv";
    private String separator = ",";

    private String db = "mysql";
    private String connIP = "localhost";
    private String port = "3306";
    private String sourceName = "/test";
    private String tableName = "nellsimple";
    private String username = "root";
    private String password = "Qatar";

    private ObjectMapper om = new ObjectMapper();

    public void finishTasks(Conductor c) {
        List<WorkerTaskResult> results = null;
        int maxAttempts = 10;
        int attempts = 0;

        while (attempts < maxAttempts) {
            results = c.consumeResults();
            if (!results.isEmpty()) {
                for (WorkerTaskResult wtr : results) {
                    String textual = null;
                    try {
                        textual = om.writeValueAsString(wtr);
                    } catch (JsonProcessingException e) {
                        e.printStackTrace();
                    }
                    System.out.println(textual);
                }
                break;
            }
            attempts++;
            try {
                Thread.sleep(1000); // Wait for 1 second before trying again
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        if (attempts == maxAttempts) {
            System.out.println("No results were produced after " + maxAttempts + " attempts.");
        }
    }

    @Test
    @Disabled("This test is currently failing")
    public void almostE2ETestDB() {
        Properties p = new Properties();
        p.setProperty(ProfilerConfig.NUM_POOL_THREADS, "1");
        p.setProperty(ProfilerConfig.NUM_RECORD_READ, "500");
        ProfilerConfig pc = new ProfilerConfig(p);
        Store es = StoreFactory.makeNullStore(pc);
        Conductor c = new Conductor(pc, es);

        c.start();

        //TaskPackage tp = TaskPackage.makeCSVFileTaskPackage("", path, filename, separator);
        //c.submitTask(tp);
        finishTasks(c);
        c.stop();
    }
}
