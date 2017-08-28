package de.ist.wikionto.research.temp;

import com.hp.hpl.jena.query.Dataset;
import com.hp.hpl.jena.query.ResultSet;

import de.ist.wikionto.research.MyLogger;
import de.ist.wikionto.triplestore.query.QueryUtil;

public abstract class Transformation {
	protected String name;
	protected MyLogger log;
	protected TransformationManager manager;
	protected ResultSet querySolutions;

	public Transformation(TransformationManager manager,String name) {
		super();
		this.manager = manager;
		this.name = name;
		this.log = new MyLogger("logs/", this.name);

	}

	public abstract void transform();;

	public ResultSet query(Dataset dataset, String queryFile) {
		ResultSet rs = QueryUtil.executeQuery(dataset, queryFile);
		return rs;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

}