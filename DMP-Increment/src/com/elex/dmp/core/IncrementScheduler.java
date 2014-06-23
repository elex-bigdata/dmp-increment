package com.elex.dmp.core;

import java.util.concurrent.atomic.AtomicInteger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.elex.dmp.utils.HdfsUtils;
import com.elex.dmp.utils.PropertiesUtil;
import com.elex.dmp.vectorizer.TFVectorsUseFixedDictionary;

public class IncrementScheduler {

	private static final Logger log = LoggerFactory.getLogger(IncrementScheduler.class);
	


	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		AtomicInteger currentPhase = new AtomicInteger();
		String[] stageArgs = {otherArgs[0],otherArgs[1]};
		String[] userFeatureArgs = {PropertiesUtil.getTimeSpanMin(),PropertiesUtil.getUserFeatureOut()};
		String[] vectorArgs = {PropertiesUtil.getUserFeatureOut(),PropertiesUtil.getVectorOut()};
		String[] userClassifyArgs = {PropertiesUtil.getBackUpDir()+"/model",PropertiesUtil.getBackUpDir()+"/dict/dictionary.file-0",
				PropertiesUtil.getBackUpDir()+"/dump",PropertiesUtil.getVectorOut()+"/tf-vectors"};
				
		int success = 0;		
		
		//stage 0
		
		if (shouldRunNextPhase(stageArgs, currentPhase)) {
			log.info("开始增量提取用户特征文件！");
			success = userFeatureExtract(userFeatureArgs);
			if (success != 0) {
				log.error("增量提取用户特征文件出错，系统退出！");
				System.exit(success);
			}
			log.info("结束增量提取用特征文件！");
		}
		//stage 1		
		if (shouldRunNextPhase(stageArgs, currentPhase)){
			log.info("使用上次全量训练生成的词典将用户特征文件进行向量转换开始！");
			success = genTFIDFVector(vectorArgs);
			if(success != 0){
				log.error("用户特征文件向量转换失败，系统退出！");
				System.exit(success);
			}
			log.info("用户特征文件向量转换正常结束！");
		}
		
		//stage 2		
		if (shouldRunNextPhase(stageArgs, currentPhase)) {
			log.info("开始用户分类并更新已有用户的分类信息！");
			success = UserClassifyAndUpdate(userClassifyArgs);
			if (success != 0) {
				log.error("更新用户分类信息失败，系统退出！");
				System.exit(success);
			}
			log.info("用户分类和更新用户分类信息正常结束！");
		}
		
		//stage 3
		if (shouldRunNextPhase(stageArgs, currentPhase)) {
			FileSystem fs = FileSystem.get(conf);			
			HdfsUtils.delFile(fs, PropertiesUtil.getVectorOut());
			HdfsUtils.delFile(fs, PropertiesUtil.getUserFeatureOut());
			fs.close();
		}
		
										
	}
	
	private static int UserClassifyAndUpdate(String[] args) throws Exception {
		
		return ToolRunner.run(new Configuration(), new UserClassifyAndUpdate(), args);
	}

	public static int userFeatureExtract(String args[]) throws Exception{		 
		 return ToolRunner.run(new Configuration(), new UserFeatureExtract(), args);
	}
	
	public static int genTFIDFVector(String args[]) throws Exception{
		String[] newArgs = {"-i", args[0], "-o", args[1], "-ow", 
				"-chunk", "128", "-wt","tf", "-s", "3", "-md",
				"2", "-x", "70", "-ng", "1", "-ml", "50", "-seq", "-n", "2"};
		return ToolRunner.run(new TFVectorsUseFixedDictionary(), newArgs);
	}
	
	
	
	
	protected static boolean shouldRunNextPhase(String[] args, AtomicInteger currentPhase) {
	    int phase = currentPhase.getAndIncrement();
	    String startPhase = args[0];
	    String endPhase = args[1];
	    boolean phaseSkipped = (startPhase != null && phase < Integer.parseInt(startPhase))
	        || (endPhase != null && phase > Integer.parseInt(endPhase));
	    if (phaseSkipped) {
	      log.info("Skipping phase {}", phase);
	    }
	    return !phaseSkipped;
	  }
	
	
}
