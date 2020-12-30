package org.pytorch.serve;
import java.util.concurrent.LinkedBlockingDeque; 
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Scheduler {
    private Logger logger = LoggerFactory.getLogger(ModelServer.class);

    private Integer counter;
    
    private Integer gpuNumber;
    private Integer modelNumber;
    LinkedBlockingDeque<SchedulerJob> lbd;

    class SchedulerJob {
        private Logger logger = LoggerFactory.getLogger(SchedulerJob.class);
    
        private String modelName;
        private String jobID;
        private Integer executedId;
    
        public SchedulerJob(Integer eid, String model_name, String jid) {
            modelName = model_name;
            jobID = jid;
            executedId = eid;
    
            logger.info("XXXXXXXXX new scheduler job {}, {}, {}", eid, model_name, jid);
        }
    }

    public Scheduler() {
        logger.info("XXXXXXXXX start the scheduler");
        lbd = new LinkedBlockingDeque<SchedulerJob>(20); 
        counter = 0;
    }

    public void SetGpuNumber(Integer n) {
        gpuNumber = n;
    }

    public void SetModelNumber(Integer n) {
        modelNumber = n;
    }

    public void addJob(String modelName, String jobId) {
        SchedulerJob job = new SchedulerJob(counter++, modelName, jobId);
        lbd.offer(job);
    }
}