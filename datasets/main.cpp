
struct Point{
    float x;
    flaot y;
    point(){}
    point(float xx,float yy):x(xx),y(yy){}
    bool operator =(point m)
    {
        if(x==m.x&&y==m.y)
        {
            return true;
        }
        return false;
    }
}
vector<float>ransac(vector<Point>points,float acc,float thresh)
{
    float scale_inner=0.5;
    int k=log(1-acc)/log(1-scale_inner*scale_inner);
    int length=points.size();
    float error_min=100000;
    float a=0;
    float b=0;
    float c=0;
    //ax+by+c=0
    
    while(k--)
    {
        int inners=0；
        float error=0;
        Point p1,p2;
        while(p1==p2)
        {
         p1(points[rand()%length].x,points[rand()%length].y);
         p2(points[rand()%length].x,points[rand()%length].y);
        }
        
        float a_temp=;
        float b_temp=;
        float c_temp=;
        for(int i=0;i<length;i++)
        {
            if(abs(a_temp*points[i].x+b_temp*points[i].y+c)<thresh)
            {
                inners++;
                error +=abs(a_temp*points[i].x+b_temp*points[i].y+c);
            }
        }
        if(inners>acc*length)
        {
            if(error/inners<error_min)
            {
                error_min=error;
                a=a_temp;
                b=b_temp;
                c=c_temp;
            }
        }
        return vector<float>{a,b,c};
    }
}