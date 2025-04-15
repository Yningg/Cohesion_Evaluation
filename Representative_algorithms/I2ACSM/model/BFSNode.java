package com.heu.cswg.model;

import java.math.BigInteger;

public class BFSNode {
	private BigInteger id;
	
	private BigInteger fId;
	
	private float nodeScore;

	public BigInteger getId() {
		return id;
	}

	public void setId(BigInteger id) {
		this.id = id;
	}

	public BigInteger getfId() {
		return fId;
	}

	public void setfId(BigInteger fId) {
		this.fId = fId;
	}

	public float getNodeScore() {
		return nodeScore;
	}

	public void setNodeScore(float nodeScore) {
		this.nodeScore = nodeScore;
	}

	@Override
	public String toString() {
		return "BFSNode [id=" + id + ", fId=" + fId + ", nodeScore="
				+ nodeScore + "]";
	}
	
	
	
}
