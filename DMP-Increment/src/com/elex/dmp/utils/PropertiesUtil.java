package com.elex.dmp.utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class PropertiesUtil {
	private static Properties pop = new Properties();
	static{
		InputStream is = null;
		try{
			is = PropertiesUtil.class.getClassLoader().getResourceAsStream("config.properties");
			pop.load(is);
		}catch(Exception e){
			e.printStackTrace();
			
		}finally{
			try {
				if(is!=null)is.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static String getTimeSpanMin(){
		return pop.getProperty("timeSpanMin");
	}
	
	public static String getUserFeatureOut(){
		return pop.getProperty("userFeatureOut");	
	}
	
	public static String getVectorOut(){
		return pop.getProperty("vectorOut");
	}
	
	public static String getClusterOut(){
		return pop.getProperty("clusterOut");
	}

	public static String getClusterMaxIter(){
		return pop.getProperty("clusterMaxIter");
	}
	
	public static String getNumTopics(){
		return pop.getProperty("num_topics");
	}
	
	public static String getDocTopicOutput(){
		return pop.getProperty("doc_topic_output");
	}
	
	public static String getCvbTemp(){
		return pop.getProperty("state_temp");
	}
	
	public static String getPercentage(){
		return pop.getProperty("percentage");
	}
	
	public static String getModelFilePath(){
		return pop.getProperty("modelFile");
	}
	
	public static String getDspUserClassFile(){
		return pop.getProperty("dspUserClassFile");
	}
	
	public static String getPoolSize(){
		return pop.getProperty("poolSize");
	}
	
	public static int getKeyWordNumber(){
		return Integer.parseInt(pop.getProperty("keyWordNumber"));
	}
	
	public static String getBackUpDir(){
		return pop.getProperty("backUpDir");
	}
	
	public static byte getType(){
		return Byte.parseByte(pop.getProperty("type"));
	}
	
}
