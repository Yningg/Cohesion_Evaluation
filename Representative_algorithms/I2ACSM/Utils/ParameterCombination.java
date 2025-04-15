package Utils;

public class ParameterCombination {
    private Integer d;
    private Integer k;
    private Float p;

    public ParameterCombination(Integer d, Integer k, Float p) {
        this.d = d;
        this.k = k;
        this.p = p;
    }

    public Integer getD() {
        return d;
    }

    public Integer getK() {
        return k;
    }

    public Float getP() {
        return p;
    }

    @Override
    public String toString() {
        return "ParameterCombination{" +
                "d=" + d +
                ", k=" + k +
                ", p=" + p +
                '}';
    }
}