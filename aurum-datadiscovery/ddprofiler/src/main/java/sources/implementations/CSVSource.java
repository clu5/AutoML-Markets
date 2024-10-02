package sources.implementations;

import au.com.bytecode.opencsv.CSVReader;
import com.codahale.metrics.Counter;
import com.codahale.metrics.MetricRegistry;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import metrics.Metrics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sources.config.CSVSourceConfig;
import sources.config.SourceConfig;
import sources.deprecated.Attribute;
import sources.deprecated.Record;
import sources.Source;
import sources.SourceType;
import sources.SourceUtils;

public class CSVSource implements Source {

    final private Logger LOG = LoggerFactory.getLogger(CSVSource.class.getName());
    private boolean useDummyColumnNames = false;

    private int tid;
    private String path;
    private String relationName;
    private CSVSourceConfig config;
    private boolean initialized = false;
    private CSVReader fileReader;
    // private TableInfo tableInfo;
    private List<Attribute> attributes;

    // metrics
    private long lineCounter = 0;
    private Counter error_records = Metrics.REG.counter((MetricRegistry.name(CSVSource.class, "error", "records")));
    private Counter success_records = Metrics.REG.counter((MetricRegistry.name(CSVSource.class, "success", "records")));

    public CSVSource() {

    }

    public CSVSource(String path, String relationName, SourceConfig config) {
	this.tid = SourceUtils.computeTaskId(path, relationName);
	this.path = path;
	this.relationName = relationName;
	this.config = (CSVSourceConfig) config;
    }

    @Override
    public String getPath() {
	return path;
    }

    @Override
    public String getRelationName() {
	return relationName;
    }

    @Override
    public SourceConfig getSourceConfig() {
	return this.config;
    }

    @Override
    public int getTaskId() {
	return tid;
    }

    @Override
    public List<Source> processSource(SourceConfig config) {
    LOG.info("Processing CSV source with config: {}", config);

    if (config == null) {
        LOG.error("SourceConfig is null");
        return new ArrayList<>();
    }

    if (!(config instanceof CSVSourceConfig)) {
        LOG.error("Expected CSVSourceConfig, but got: {}", config.getClass().getName());
        return new ArrayList<>();
    }
	assert (config instanceof CSVSourceConfig);

	this.config = (CSVSourceConfig) config;

	List<Source> tasks = new ArrayList<>();

	CSVSourceConfig csvConfig = (CSVSourceConfig) config;
	String pathToSources = csvConfig.getPath();
    LOG.info("Path to sources: {}", pathToSources);
    if (pathToSources == null || pathToSources.isEmpty()) {
        LOG.error("Path to sources is null or empty");
        return tasks;
    }

	// TODO: at this point we'll be harnessing metadata from the source

	File folder = new File(pathToSources);
    LOG.info("Absolute path to sources: {}", folder.getAbsolutePath());
    LOG.info("Directory exists: {}", folder.exists());
    LOG.info("Is a directory: {}", folder.isDirectory());
    LOG.info("Can read: {}", folder.canRead());
    if (!folder.exists() || !folder.isDirectory()) {
        LOG.error("Path does not exist or is not a directory: {}", pathToSources);
        return tasks;
    }


	int totalFiles = 0;
	int tt = 0;

	File[] filePaths = folder.listFiles();
    if (filePaths == null) {
        LOG.error("Unable to list files in directory: {}", pathToSources);
        return tasks;
    }

	for (File f : filePaths) {
	    tt++;
	    if (f.isFile()) {
		String path = f.getParent() + File.separator;
		String name = f.getName();
        LOG.info("Processing file: {}", name);
		// Make the csv config specific to the relation
		CSVSource task = new CSVSource(path, name, config);
		totalFiles++;
		// c.submitTask(pt);
		tasks.add(task);
	    }
	}

	LOG.info("Total files submitted for processing: {} - {}", totalFiles, tt);
	return tasks;
    }

    @Override
    public SourceType getSourceType() {
	return SourceType.csv;
    }


    private boolean isNumeric(String str) {
        if (str == null || str.trim().isEmpty()) {
            return false;
        }
        str = str.trim();
        try {
            Double.parseDouble(str.split("\\s+")[0]);
            return true;
        } catch(NumberFormatException e) {
            return false;
        }
    }


    @Override
    public List<Attribute> getAttributes() throws IOException, SQLException {
        String fullPath = this.path + this.relationName;
        LOG.info("Getting attributes for file: {}", fullPath);
        char separator = this.config.getSeparator().charAt(0);
        if (!initialized) {
            fileReader = new CSVReader(new FileReader(fullPath), separator);
            initialized = true;
            LOG.info("Initialized CSVReader for file: {}", fullPath);
        }
        if (attributes == null) {
            String[] firstRow = fileReader.readNext();
            lineCounter++;
            LOG.info("Read first row: {}", Arrays.toString(firstRow));

            List<Attribute> attrList = new ArrayList<>();
            useDummyColumnNames = false;

            for (int i = 0; i < firstRow.length; i++) {
                String attrName;
                if (isNumeric(firstRow[i].trim())) {
                    // This is likely data, not a header. Create a dummy column name.
                    attrName = "Column_" + (i + 1);
                    useDummyColumnNames = true;
                    LOG.info("Using dummy name for column {}: {}", i, attrName);

                } else {
                    attrName = firstRow[i];
                    LOG.info("Using original name for column {}: {}", i, attrName);
                }
                Attribute attr = new Attribute(attrName);
                attrList.add(attr);
            }
            this.attributes = attrList;
            if (useDummyColumnNames) {
                LOG.info("Using dummy column names for file: {}", fullPath);
                // Reset the file reader to start from the beginning
                fileReader.close();
                fileReader = new CSVReader(new FileReader(fullPath), separator);
                lineCounter = 0;
                LOG.info("Reset CSVReader to start of file");
            }
        }
        return attributes;
    }
    private void addRowToData(Map<Attribute, List<String>> data, String[] values) {
        addRowToData(data, Arrays.asList(values));
    }


    private void addRowToData(Map<Attribute, List<String>> data, List<String> values) {
        if (values.size() != data.size()) {
            error_records.inc();
            return; // Some error while parsing data, a row has a different format
        }
        success_records.inc();
        int currentIdx = 0;
        for (List<String> vals : data.values()) { // ordered iteration
            vals.add(values.get(currentIdx));
            currentIdx++;
        }
    }

    @Override
    public Map<Attribute, List<String>> readRows(int num) throws IOException, SQLException {
        String fullPath = this.path + this.relationName;
        char separator = this.config.getSeparator().charAt(0);

        if (!initialized) {
            fileReader = new CSVReader(new FileReader(fullPath), separator);
            initialized = true;
        }

        Map<Attribute, List<String>> data = new LinkedHashMap<>();
        // Make sure attrs is populated, if not, populate it here
        if (data.isEmpty()) {
            List<Attribute> attrs = this.getAttributes();
            attrs.forEach(a -> data.put(a, new ArrayList<>()));
        }
        // If we're using dummy column names, we need to start reading from the first row
        if (useDummyColumnNames && lineCounter == 0) {
            // The first row is data, not headers
            String[] firstDataRow = fileReader.readNext();
            lineCounter++;
            if (firstDataRow != null) {
                addRowToData(data, Arrays.asList(firstDataRow));
                num--; // Decrease num as we've already read one row
            }
        }

        // Read data and insert in order
        List<Record> recs = new ArrayList<>();
        boolean readData = this.read(num, recs);
        if (!readData && recs.isEmpty()) {
            return null;
        }

        for (Record r : recs) {
                addRowToData(data, r.getTuples());
            }
        return data;
        }

    private boolean read(int numRecords, List<Record> rec_list) throws IOException {
        boolean read_lines = false;
        String[] res = null;
        for (int i = 0; i < numRecords && (res = fileReader.readNext()) != null; i++) {
            lineCounter++;
            read_lines = true;
            Record rec = new Record();
            rec.setTuples(res);
            rec_list.add(rec);
        }
        return read_lines;
    }

    @Override
    public void close() {
	try {
	    fileReader.close();
	} catch (IOException e) {
	    e.printStackTrace();
	}
    }

}
