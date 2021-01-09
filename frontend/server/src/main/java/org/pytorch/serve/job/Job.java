package org.pytorch.serve.job;

import java.util.Map;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class Job {

    private String modelName;
    private String modelVersion;
    private WorkerCommands cmd; // Else its data msg or inf requests
    private RequestInput input;
    private long begin;
    private long scheduled;
    private long deadline;

    private static final Logger logger = LoggerFactory.getLogger(Job.class);

    public Job(String modelName, String version, WorkerCommands cmd, RequestInput input, long deadlineTime) {
        this.modelName = modelName;
        this.cmd = cmd;
        this.input = input;
        this.modelVersion = version;
        begin = System.nanoTime();
        scheduled = begin;
        deadline = begin + deadlineTime;
	    logger.info("XXXXXXXXXXXXXXXXXXXXXXX  NEW JOB {} {}, begin: {}, deadline: {}", cmd, modelName, begin, deadline);
    }

    public long getDeadline() {
        return deadline;
    }

    public String getJobId() {
        return input.getRequestId();
    }

    public String getModelName() {
        return modelName;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public WorkerCommands getCmd() {
        return cmd;
    }

    public boolean isControlCmd() {
        return !WorkerCommands.PREDICT.equals(cmd);
    }

    public RequestInput getPayload() {
        return input;
    }

    public void setScheduled() {
        scheduled = System.nanoTime();
    }

    public long getBegin() {
        return begin;
    }

    public long getScheduled() {
        return scheduled;
    }

    public abstract void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders);

    public abstract void sendError(int status, String error);
}
