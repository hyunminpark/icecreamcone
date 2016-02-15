  function y = cute2(x)
  % cute function can be of arbitrary size
  % a good start is with "ones"
  % example usage: xout=fminunc(@cute,ones(100,1),[])
  % example usage: [xout,iteratesGradNorms]=newtonLikeMethod(@cute_wrap,-ones(1000,1),1,1e-9,100)
  % example usage: [xc,histout,costdata]=cgtrust(-ones(1000,1),@cute_fg,1e-10)
    N = length(x);
    I = 2:N-3;
    y = sum( (-4*x(I)+3.0).^2 ) + sum( ( x(I).^2 + 2*x(I+1).^2 + ...
      3*x(I+2).^2 + 4*x(I+3).^2 + 5*x(1).^2 ).^2 ) ;
